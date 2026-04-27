"""DataVault — encrypted, content-addressed binary store for face assets.

Architecture
------------
All sensitive face images (portraits, detection evidence, enrollment photos)
are stored as opaque `.bin` files whose names are the SHA-256 hash of the
*plaintext* JPEG bytes.  No filename ever contains an entity ID or person ID.

Encryption
----------
Each blob is encrypted with **XChaCha20-Poly1305** (256-bit, authenticated).
The blob wire format is::

    [1B version=0x01][1B algo=0x01][24B nonce][ciphertext + 16B Poly1305 tag]

The encryption key is loaded from the environment variable ``TRACE_VAULT_KEY``
(64 hex chars = 32 bytes).  If the variable is absent or all-zero the vault
operates in *passthrough mode* — bytes are stored unencrypted (dev/CI use only).

Directory layout::

    data/vault/
        portraits/{sha256[:2]}/{sha256}.bin
        evidence/{YYYY-MM-DD}/{sha256[:2]}/{sha256}.bin
        enrollment/{sha256[:2]}/{sha256}.bin
    data/index/
        portraits.json   — { entity_id → { key, score, updated_at } }
        evidence.json    — { detection_id → { key, entity_id, ts } }
        enrollment.json  — { person_id → [sha256, ...] }

Thread safety
-------------
* Per-entity locks for portrait writes.
* Module-level lock for all three JSON index files (atomic write via temp file).
* Evidence writes are append-only and use the entity-level lock.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from loguru import logger

from trace_aml.core.config import Settings

# ── Blob format constants ────────────────────────────────────────────────────
_VERSION_1      : int = 0x01
_ALGO_CHACHA20  : int = 0x01   # ChaCha20-Poly1305 (IETF, 12-byte nonce, 256-bit key)
_NONCE_SIZE     : int = 12     # IETF ChaCha20-Poly1305 uses 12-byte (96-bit) nonce
_TAG_SIZE       : int = 16     # Poly1305 authentication tag
_HEADER_SIZE    : int = 2      # version + algo_id bytes before nonce
_MIN_BLOB_SIZE  : int = _HEADER_SIZE + _NONCE_SIZE + _TAG_SIZE  # 30 bytes minimum


# ── Encryption / decryption ──────────────────────────────────────────────────

def _load_key() -> bytes | None:
    """Load vault key from OS keychain (Electron) or environment variable (dev).

    Priority order:
    1. OS keychain via ``keyring`` — used when packaged as Electron app.
       The Electron main process stores the key on first launch using:
       ``keyring.set_password("trace-aml", "vault_key", hex_key)``
    2. ``TRACE_VAULT_KEY`` environment variable — used in dev / browser mode
       via ``.env`` file loaded by python-dotenv (or set directly in shell).
    3. Neither set → passthrough mode (no encryption, dev/CI use only).

    Returns 32-byte key, or None if key is absent/zero (passthrough mode).
    """
    raw: str = ""

    # 1. Try OS keychain first (Electron packaging)
    try:
        import keyring as _kr  # type: ignore
        stored = _kr.get_password("trace-aml", "vault_key")
        if stored:
            raw = stored.strip()
    except Exception:
        pass  # keyring not installed or platform not supported — fall through

    # 2. Fall back to environment variable (browser dev / .env file)
    if not raw:
        raw = os.environ.get("TRACE_VAULT_KEY", "").strip()

    if not raw:
        return None

    try:
        key = bytes.fromhex(raw)
    except ValueError:
        logger.error("DataVault: vault key is not valid hex — vault in passthrough mode")
        return None
    if len(key) != 32:
        logger.error(
            "DataVault: vault key must be 64 hex chars (32 bytes), got {} — passthrough mode",
            len(key) * 2,
        )
        return None
    if all(b == 0 for b in key):
        return None  # all-zero key = explicit passthrough
    return key


def _encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Return versioned encrypted blob: header + nonce + ciphertext+tag.

    Uses ChaCha20-Poly1305 (IETF, RFC 8439) with a 12-byte random nonce.
    Wire format: [version=0x01][algo=0x01][nonce 12B][ciphertext + Poly1305 tag 16B]
    """
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305  # type: ignore
    nonce = os.urandom(_NONCE_SIZE)  # 12 bytes = 96-bit random nonce
    cipher = ChaCha20Poly1305(key)
    ciphertext = cipher.encrypt(nonce, plaintext, None)  # includes 16-byte tag
    return bytes([_VERSION_1, _ALGO_CHACHA20]) + nonce + ciphertext


def _decrypt(blob: bytes, key: bytes) -> bytes | None:
    """Decrypt a versioned blob. Returns None on auth failure or unknown format."""
    if len(blob) < _MIN_BLOB_SIZE:
        logger.warning("DataVault: blob too short ({} bytes) — skipping", len(blob))
        return None
    version = blob[0]
    algo_id = blob[1]
    if version != _VERSION_1 or algo_id != _ALGO_CHACHA20:
        logger.warning(
            "DataVault: unknown blob format v={} algo={} — cannot decrypt", version, algo_id
        )
        return None
    nonce = blob[_HEADER_SIZE : _HEADER_SIZE + _NONCE_SIZE]
    ciphertext = blob[_HEADER_SIZE + _NONCE_SIZE :]
    try:
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305  # type: ignore
        cipher = ChaCha20Poly1305(key)
        return cipher.decrypt(nonce, ciphertext, None)  # raises InvalidTag if auth fails
    except Exception:
        # InvalidTag = wrong key OR tampered file — both handled as silent None
        logger.warning("DataVault: authentication failure (wrong key or tampered blob)")
        return None


# ── Index atomic write helper ────────────────────────────────────────────────

def _atomic_json_write(path: Path, data: dict) -> None:
    """Write JSON to a temp file then atomically rename over target.

    os.replace() is atomic on POSIX and on Windows NTFS so a crash midway
    will never leave a half-written index.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _load_json_index(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("DataVault: failed to load index {} — {}", path.name, exc)
        return {}


# ── DataVault ────────────────────────────────────────────────────────────────

class DataVault:
    """Encrypted, content-addressed binary store for all face image assets.

    Callers never receive filesystem paths.  All I/O uses logical keys
    (entity_id, detection_id, person_id).  The vault handles path resolution,
    encryption, decryption, and index maintenance internally.
    """

    def __init__(self, settings: Settings) -> None:
        vault_root = Path(settings.vault.portraits_dir).parent  # data/vault
        self._portraits_dir  = Path(settings.vault.portraits_dir)
        self._evidence_dir   = Path(settings.vault.evidence_dir)
        self._enrollment_dir = Path(settings.vault.enrollment_dir)
        self._index_dir      = Path(settings.vault.index_dir)

        for d in (self._portraits_dir, self._evidence_dir,
                  self._enrollment_dir, self._index_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._portraits_idx_path  = self._index_dir / "portraits.json"
        self._evidence_idx_path   = self._index_dir / "evidence.json"
        self._enrollment_idx_path = self._index_dir / "enrollment.json"

        # Load indexes into memory (fast O(1) lookups thereafter)
        self._portraits_idx:  dict = _load_json_index(self._portraits_idx_path)
        self._evidence_idx:   dict = _load_json_index(self._evidence_idx_path)
        self._enrollment_idx: dict = _load_json_index(self._enrollment_idx_path)

        # Thread safety
        self._index_lock    = threading.Lock()   # protects all three index dicts + files
        self._entity_locks: dict[str, threading.Lock] = {}
        self._entity_meta_lock = threading.Lock()

        # Encryption key (None = passthrough mode)
        self._key = _load_key()
        if self._key is None:
            logger.warning(
                "DataVault: TRACE_VAULT_KEY not set — running in PASSTHROUGH mode "
                "(images stored unencrypted). Set the key for production use."
            )
        else:
            logger.info("DataVault: XChaCha20-Poly1305 encryption enabled.")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _entity_lock(self, entity_id: str) -> threading.Lock:
        with self._entity_meta_lock:
            if entity_id not in self._entity_locks:
                self._entity_locks[entity_id] = threading.Lock()
            return self._entity_locks[entity_id]

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _blob_path(self, base_dir: Path, sha: str, date_prefix: str | None = None) -> Path:
        if date_prefix:
            return base_dir / date_prefix / sha[:2] / f"{sha}.bin"
        return base_dir / sha[:2] / f"{sha}.bin"

    def _write_blob(self, path: Path, plaintext: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._key is not None:
            data = _encrypt(plaintext, self._key)
        else:
            data = plaintext  # passthrough
        path.write_bytes(data)

    def _read_blob(self, path: Path) -> bytes | None:
        if not path.exists():
            return None
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.warning("DataVault: cannot read blob {} — {}", path.name, exc)
            return None
        if self._key is not None:
            return _decrypt(data, self._key)
        return data  # passthrough

    # ── Portrait API ─────────────────────────────────────────────────────────

    def put_portrait(self, entity_id: str, jpeg_bytes: bytes, score: float) -> str:
        """Encrypt and store portrait for entity_id.

        Returns the SHA-256 content key (hex string).
        Thread-safe per entity — concurrent updates for different entities
        never block each other.
        """
        sha = self._sha256(jpeg_bytes)
        lock = self._entity_lock(entity_id)
        with lock:
            path = self._blob_path(self._portraits_dir, sha)
            self._write_blob(path, jpeg_bytes)
            with self._index_lock:
                self._portraits_idx[entity_id] = {
                    "key": sha,
                    "score": round(float(score), 6),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                _atomic_json_write(self._portraits_idx_path, self._portraits_idx)
        logger.debug("DataVault: portrait stored for {} (sha={}…)", entity_id, sha[:8])
        return sha

    def get_portrait_bytes(self, entity_id: str) -> bytes | None:
        """Return decrypted JPEG bytes for entity's best portrait, or None."""
        with self._index_lock:
            entry = self._portraits_idx.get(entity_id)
        if not entry:
            return None
        sha = entry["key"]
        path = self._blob_path(self._portraits_dir, sha)
        result = self._read_blob(path)
        if result is None:
            logger.debug("DataVault: portrait blob missing/unreadable for {}", entity_id)
        return result

    def has_portrait(self, entity_id: str) -> bool:
        """O(1) existence check — reads index only."""
        with self._index_lock:
            return entity_id in self._portraits_idx

    def get_portrait_score(self, entity_id: str) -> float | None:
        """Return stored cosine similarity score, or None."""
        with self._index_lock:
            entry = self._portraits_idx.get(entity_id)
        return float(entry["score"]) if entry else None

    def delete_portrait(self, entity_id: str) -> None:
        """Delete portrait blob and remove from index."""
        lock = self._entity_lock(entity_id)
        with lock:
            with self._index_lock:
                entry = self._portraits_idx.pop(entity_id, None)
                if entry:
                    _atomic_json_write(self._portraits_idx_path, self._portraits_idx)
            if entry:
                path = self._blob_path(self._portraits_dir, entry["key"])
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("DataVault: could not delete portrait blob — {}", exc)
                logger.debug("DataVault: portrait deleted for {}", entity_id)

    # ── Evidence API ─────────────────────────────────────────────────────────

    def put_evidence(
        self,
        detection_id: str,
        entity_id: str,
        jpeg_bytes: bytes,
        timestamp: str | None = None,
    ) -> str:
        """Encrypt and store a detection screenshot.

        Returns the SHA-256 content key.
        """
        sha = self._sha256(jpeg_bytes)
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        date_str = ts[:10]  # YYYY-MM-DD
        path = self._blob_path(self._evidence_dir, sha, date_prefix=date_str)
        self._write_blob(path, jpeg_bytes)
        with self._index_lock:
            self._evidence_idx[detection_id] = {
                "key": sha,
                "entity_id": entity_id,
                "ts": ts,
                "date": date_str,
            }
            _atomic_json_write(self._evidence_idx_path, self._evidence_idx)
        logger.debug(
            "DataVault: evidence stored for detection {} entity {} (sha={}…)",
            detection_id, entity_id, sha[:8],
        )
        return sha

    def get_evidence_bytes(self, detection_id: str) -> bytes | None:
        """Return decrypted JPEG bytes for a detection screenshot, or None."""
        with self._index_lock:
            entry = self._evidence_idx.get(detection_id)
        if not entry:
            return None
        sha = entry["key"]
        date_str = entry.get("date", "")
        path = self._blob_path(self._evidence_dir, sha, date_prefix=date_str)
        return self._read_blob(path)

    def has_evidence(self, detection_id: str) -> bool:
        """O(1) existence check from evidence index."""
        with self._index_lock:
            return detection_id in self._evidence_idx

    # ── Enrollment API ───────────────────────────────────────────────────────

    def put_enrollment_image(self, person_id: str, jpeg_bytes: bytes) -> str:
        """Encrypt and store one enrollment image. Returns SHA-256 key."""
        sha = self._sha256(jpeg_bytes)
        path = self._blob_path(self._enrollment_dir, sha)
        self._write_blob(path, jpeg_bytes)
        with self._index_lock:
            existing = self._enrollment_idx.get(person_id, [])
            if sha not in existing:
                existing.append(sha)
            self._enrollment_idx[person_id] = existing
            _atomic_json_write(self._enrollment_idx_path, self._enrollment_idx)
        logger.debug("DataVault: enrollment image stored for {} (sha={}…)", person_id, sha[:8])
        return sha

    def get_enrollment_image_bytes(self, person_id: str) -> list[bytes]:
        """Return list of decrypted JPEG bytes for all enrollment images."""
        with self._index_lock:
            keys = list(self._enrollment_idx.get(person_id, []))
        result: list[bytes] = []
        for sha in keys:
            path = self._blob_path(self._enrollment_dir, sha)
            data = self._read_blob(path)
            if data is not None:
                result.append(data)
        return result

    @contextlib.contextmanager
    def extract_enrollment_to_tempdir(
        self, person_id: str
    ) -> Generator[Path, None, None]:
        """Context manager: decrypt enrollment images to a temp directory.

        Yields a Path to the temp dir containing ``{n:03d}.jpg`` files.
        The temp dir is deleted on exit regardless of exceptions.

        Usage::

            with vault.extract_enrollment_to_tempdir(person_id) as tmp:
                for img_path in sorted(tmp.iterdir()):
                    img = cv2.imread(str(img_path))
                    ...
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="trace_vault_tmp_"))
        try:
            images = self.get_enrollment_image_bytes(person_id)
            for i, jpeg_bytes in enumerate(images):
                out = tmp_dir / f"{i:03d}.jpg"
                out.write_bytes(jpeg_bytes)
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def delete_person_enrollment(self, person_id: str) -> None:
        """Delete all enrollment blobs for a person."""
        with self._index_lock:
            keys = self._enrollment_idx.pop(person_id, [])
            if keys:
                _atomic_json_write(self._enrollment_idx_path, self._enrollment_idx)
        for sha in keys:
            path = self._blob_path(self._enrollment_dir, sha)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if keys:
            logger.debug("DataVault: deleted {} enrollment blobs for {}", len(keys), person_id)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def wipe(self) -> None:
        """Delete ALL vault blobs and index files. Called during factory reset."""
        for d in (self._portraits_dir, self._evidence_dir, self._enrollment_dir):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
        with self._index_lock:
            self._portraits_idx.clear()
            self._evidence_idx.clear()
            self._enrollment_idx.clear()
            for p in (
                self._portraits_idx_path,
                self._evidence_idx_path,
                self._enrollment_idx_path,
            ):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        logger.info("DataVault: all blobs and indexes wiped.")

    def stats(self) -> dict[str, int]:
        """Return blob counts per namespace."""
        with self._index_lock:
            return {
                "portraits": len(self._portraits_idx),
                "evidence": len(self._evidence_idx),
                "enrollment": sum(len(v) for v in self._enrollment_idx.values()),
                "persons_enrolled": len(self._enrollment_idx),
            }

    def rebuild_index(self) -> dict[str, int]:
        """Scan vault dirs and rebuild evidence index from filenames.

        Portraits and enrollment indexes cannot be fully rebuilt from blobs
        alone (the entity_id → sha mapping is not stored inside the blob).
        This method only rebuilds what it can from the filesystem.
        """
        counts = {"evidence_scanned": 0}
        new_evidence: dict = {}
        for blob in self._evidence_dir.rglob("*.bin"):
            sha = blob.stem
            date_str = blob.parts[-3] if len(blob.parts) >= 3 else ""
            new_evidence[sha] = {"key": sha, "entity_id": "unknown", "ts": "", "date": date_str}
            counts["evidence_scanned"] += 1
        with self._index_lock:
            self._evidence_idx.update(new_evidence)
            _atomic_json_write(self._evidence_idx_path, self._evidence_idx)
        logger.info("DataVault: index rebuilt — {}", counts)
        return counts
