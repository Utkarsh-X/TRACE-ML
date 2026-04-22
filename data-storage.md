# TRACE-AML: Data Storage — Refined Architecture & Implementation Plan

> **Status**: Revised after review session. This is the final agreed direction.

---

## 1. Honest Audit — What Each Subsystem Looks Like Today

| Subsystem | Location | Verdict | Priority |
|---|---|---|---|
| **Portraits** | `data/portraits/PRC001.jpg` | ❌ Filename = entity ID. Browsable face database. | **HIGH** |
| **Screenshots** | `data/screenshots/DET-*.jpg` | ⚠️ Browsable, but kept intentionally (PDF gallery). Naming is opaque enough. Need to move into vault. | **HIGH** |
| **Person images** | `data/person_images/PRC001/upload_001.jpg` | ❌ Folder name = person ID. Fully readable enrolment archive. | **MEDIUM** |
| **Vectors** | `data/vectors/*.lance` | ✅ Arrow binary format, already opaque. No changes needed. | NONE |
| **Exports** | `data/exports/` | ⚠️ PDFs contain identity + face crops. OS-level ACL is the main protection needed. | LOW |
| **Logs** | `data/logs/` | ⚠️ Paths and entity IDs in log lines. Acceptable for operator logs. | LOW |

---

## 2. Core Problems — Revised Assessment

### ❌ Problem 1 — Identity name leakage in filenames (portraits + person_images)
```
data/portraits/PRC001.jpg       ← entity ID is the filename
data/person_images/PRC001/      ← person ID is the folder name
```
Anyone who opens `data/` in Windows Explorer sees a directory of faces with their
system IDs clearly readable. This is the **primary** thing to fix.

### ❌ Problem 2 — Images are directly viewable (JPEG stays JPEG)
Even after renaming to `.bin`, the bytes inside are still valid JPEG — any
hex editor or image viewer that ignores the extension will show the face.
The user's requirement: images must **not be directly visible** to the end user,
only to the application. This needs a lightweight obfuscation layer.

### ❌ Problem 3 — Absolute OS paths stored in LanceDB
```
DetectionEvent.screenshot_path = "D:\\github FORK\\TRACE-ML\\data\\screenshots\\DET-....jpg"
```
If the `data/` directory is ever moved, renamed, or the system migrated to another
machine, every path reference in every detection record breaks. The PDF gallery
algorithm already handles this gracefully (skips non-existent paths), but it's
fragile and wrong in principle.

### ✅ Problem 4 — Screenshot accumulation → NOT a problem
The user has confirmed screenshots are **intentionally kept** — they are the source
material for the PDF evidence gallery. The existing throttle (5 s/entity, face-context
crop, 320px JPEG-80) already keeps per-file sizes small (~10–15 KB). No deletion
necessary. We keep all of them.

### ✅ Problem 5 — Gallery selection algorithm → ALREADY EXISTS
The PDF handler at `src/trace_aml/actions/pdf_handler.py:_select_gallery_shots()`
**already implements temporal bucketing**: it picks the highest-confidence detection
per 10-minute window (max 12 shots, 3-per-row). This is solid.

The one improvement worth adding: use a **composite quality score** instead of
confidence alone, so the gallery prefers sharper, better-lit frames even if their
gallery match confidence is slightly lower.

---

## 3. The Obfuscation Decision — XOR vs AES

The user's constraint: **maximum protection, minimal CPU overhead**.

| Scheme | CPU cost per 15 KB image | Readable without key? | Complexity |
|---|---|---|---|
| Rename to `.bin` only | ~0 | Still yes (bytes are valid JPEG) | Trivial |
| **XOR with 32-byte key** | **~0.05 ms** | **No — bytes look random** | **Minimal** |
| AES-128-CBC (Fernet) | ~1–2 ms | No — cryptographically secure | Moderate |
| AES-256-GCM | ~2–5 ms | No — max security | Higher |

**Decision: XOR with a 32-byte key stored in `.env`**

Rationale:
- At 15 KB per screenshot, XOR processes ~15,000 bytes in < 0.1 ms — imperceptible
- The key is only stored in `.env` (already gitignored). Without `.env`, all `.bin`
  files look like random noise to any viewer
- Double-clicking a `.bin` file in Windows Explorer shows garbled bytes, not a face
- The design is **directly upgradeable**: replace the XOR function with
  `cryptography.fernet.Fernet` and nothing else changes — same API, same vault,
  same index. This is the upgrade path if the deployment ever needs AES-128
- XOR is symmetric and in-place: `scramble(scramble(data)) == data`, so read/write
  use the exact same function

```python
# Implementation (from .env: TRACE_VAULT_KEY=64-hex-char-string)
_KEY = bytes.fromhex(os.environ.get("TRACE_VAULT_KEY", "00" * 32))

def _scramble(data: bytes) -> bytes:
    """XOR each byte against cycling 32-byte key. Near-zero CPU overhead."""
    if all(b == 0 for b in _KEY):  # key not set = dev mode (passthrough)
        return data
    key = _KEY
    return bytes(b ^ key[i % 32] for i, b in enumerate(data))

# _unscramble is identical — XOR is its own inverse
_unscramble = _scramble
```

> [!IMPORTANT]
> If `TRACE_VAULT_KEY` is not set in `.env`, the vault operates in **passthrough mode**
> (no obfuscation, just renamed to `.bin`). This means development machines without
> the key still work. Production deployments set the key once and never change it.

> [!CAUTION]
> XOR is **obfuscation, not encryption**. A determined attacker with the file AND the
> key can recover the image trivially. The threat model here is: casual OS browser
> access, accidental disclosure via backup copy, or a non-privileged user browsing the
> `data/` folder. For classified / law-enforcement deployments, upgrade to Fernet AES.

---

## 4. Proposed Directory Layout

```
data/
  vault/
    portraits/
      {h0}{h1}/             ← first 2 chars of sha256 (256 subdirs max)
        {sha256}.bin        ← XOR-scrambled JPEG bytes
    evidence/
      {YYYY-MM-DD}/         ← date-partitioned for easy future archival
        {h0}{h1}/
          {sha256}.bin
    enrollment/
      {h0}{h1}/
        {sha256}.bin
  index/
    portraits.json          ← { "PRC001": "3fa2...c8", "UNK-001": "9b1f...44" }
    evidence.json           ← { "DET-abc123": { "key": "...", "ts": "...", "entity_id": "..." } }
    enrollment.json         ← { "PRC001": ["sha256_a", "sha256_b", ...] }
  vectors/         ← UNCHANGED (already opaque LanceDB format)
  exports/         ← UNCHANGED (PDFs — OS ACL protection)
  logs/            ← UNCHANGED
```

**What the OS file browser shows now:**
```
data/vault/portraits/3f/3fa2c8b14d...bin    ← means nothing
data/vault/evidence/2026-04-22/9b/9b1f...bin
```
Compare to today:
```
data/portraits/PRC001.jpg                   ← "Person PRC001's face"
data/screenshots/DET-20260419T082742Z.jpg   ← "Detection at 8:27 AM"
```

---

## 5. The DataVault Module

**File to create:** `src/trace_aml/store/data_vault.py`

This is the **single point of truth** for all image I/O. No code outside this module
reads from or writes to the vault directories directly.

```python
class DataVault:
    """
    Content-addressed, XOR-obfuscated binary store for face asset images.
    
    All callers use logical keys (entity_id, detection_id, person_id).
    The vault resolves these → SHA-256 blob keys → filesystem paths internally.
    Filesystem paths are NEVER returned to callers.
    """
    
    # ── Portrait API ────────────────────────────────────────────────
    def put_portrait(self, entity_id: str, jpeg_bytes: bytes, score: float) -> str:
        """Store or replace portrait. Returns sha256 key."""
    
    def get_portrait_bytes(self, entity_id: str) -> bytes | None:
        """Return raw JPEG bytes (after unscrambling), or None."""
    
    def has_portrait(self, entity_id: str) -> bool:
        """Fast existence check without reading blob."""
    
    def get_portrait_score(self, entity_id: str) -> float | None:
        """Return stored similarity score for gate decisions."""
    
    def delete_portrait(self, entity_id: str) -> None:
        """Remove portrait + update index."""
    
    # ── Evidence (screenshot) API ───────────────────────────────────
    def put_evidence(self, detection_id: str, entity_id: str, jpeg_bytes: bytes) -> str:
        """Store a detection screenshot. Returns sha256 key."""
    
    def get_evidence_bytes(self, detection_id: str) -> bytes | None:
        """Return raw JPEG bytes for this detection_id, or None."""
    
    def get_evidence_path_for_pdf(self, detection_id: str) -> str | None:
        """
        Special method only for the PDF handler.
        Writes evidence to a temp file, returns its path for base64 encoding.
        The temp file is deleted after the PDF is generated.
        """
    
    # ── Enrollment image API ────────────────────────────────────────
    def put_enrollment_image(self, person_id: str, jpeg_bytes: bytes) -> str:
        """Store one enrollment image. Returns sha256 key."""
    
    def get_enrollment_images(self, person_id: str) -> list[bytes]:
        """Return all enrollment JPEG images for a person."""
    
    def delete_person(self, person_id: str) -> None:
        """Remove all enrollment images and portrait for this person."""
```

---

## 6. Fixing Absolute Paths in LanceDB

**Current state:**
```python
DetectionEvent.screenshot_path = "D:\\github FORK\\TRACE-ML\\data\\screenshots\\DET-abc.jpg"
```

**After migration:**
```python
DetectionEvent.screenshot_path = "vault:evidence:DET-abc123"   # logical vault key
# OR simpler — reuse detection_id (it's already unique):
# screenshot_path stores empty string; evidence is looked up by detection_id from vault
```

The cleanest approach: **make `screenshot_path` store a vault key prefix** (`vault:{sha256}`).
The PDF handler's `_b64_image()` function is updated to detect this prefix and route
through `vault.get_evidence_bytes()` instead of opening a file path.

This means:
- Old records with absolute paths: `_b64_image` falls back gracefully (file won't exist
  on new machine, returns `None`, gallery shows placeholder — acceptable)
- New records: use vault keys, work anywhere the `data/vault/` directory exists

---

## 7. Gallery Algorithm — Current State & Enhancement

### What already exists (keep it)

`pdf_handler.py:_select_gallery_shots()` — **already well-implemented**:
- Sorts detections chronologically
- Divides timeline into 10-minute buckets
- Picks highest `confidence` detection per bucket
- Returns max 12 images in chronological order → 3-per-row grid in PDF

This is the right algorithm. The only gap:

### Proposed enhancement — composite quality scoring

Currently the bucket winner is chosen by `confidence` (gallery match score) alone.
A better metric is a **composite score** combining face quality AND match confidence:

```python
composite = 0.6 * smoothed_confidence + 0.4 * face_quality_from_metadata
```

Why: a face detected at 78% confidence but low sharpness will produce a blurry
thumb in the PDF. A 71% confidence frame that is perfectly sharp is a better exhibit.
This 2-line change makes the gallery visibly more professional.

---

## 8. Implementation Plan — Prioritized

### Phase 1 — Foundation (implement first, ~2 days)

**Files to create:**
- `src/trace_aml/store/data_vault.py` — the full DataVault class

**Files to modify:**
- `config/config.demo.yaml` — add `vault.obfuscation_enabled` + key reference
- `src/trace_aml/core/config.py` — add `VaultSettings` model
- `.env.example` — add `TRACE_VAULT_KEY=` documentation entry

**What this achieves:**
- DataVault exists, can be tested in isolation
- XOR scramble/unscramble is working
- Index files are being maintained
- No other code changed yet

---

### Phase 2 — Portrait Migration (~1 day)

**Files to modify:**
- `src/trace_aml/store/portrait_store.py` — delegate storage to DataVault internally
  (keep the public API identical — `try_update_portrait`, `get_portrait_path*`,
  `delete_portrait` — so zero changes cascade to `session.py`)
- `src/trace_aml/service/app.py` — portrait GET endpoint: switch from `FileResponse`
  to `StreamingResponse(BytesIO(vault.get_portrait_bytes(entity_id)))`
- `src/trace_aml/service/app.py` — portrait POST (upload) endpoint: route bytes through vault

**Migration script to create:**
- `scripts/migrate_portraits_to_vault.py` — reads existing `data/portraits/*.jpg`,
  writes into vault layout, updates `portraits.json` index, deletes originals

---

### Phase 3 — Evidence (Screenshot) Migration (~1 day)

**Files to modify:**
- `src/trace_aml/pipeline/session.py:_save_detection()` — route screenshot bytes
  through `vault.put_evidence()` instead of writing to `screenshots/` directly;
  store `vault:{sha256}` in `DetectionEvent.screenshot_path`
- `src/trace_aml/actions/pdf_handler.py:_b64_image()` — detect `vault:` prefix and
  call `vault.get_evidence_bytes()` (falls back to file path for legacy records)
- `src/trace_aml/actions/pdf_handler.py:_select_gallery_shots()` — add composite
  quality score for bucket winner selection

---

### Phase 4 — Enrollment Image Migration (~half day)

**Files to modify:**
- `src/trace_aml/pipeline/collect.py:person_image_dir()` → `vault.put_enrollment_image()`
- `src/trace_aml/pipeline/train.py` → `vault.get_enrollment_images()`
- `src/trace_aml/service/person_api.py` — upload endpoint → vault

**Migration script to create:**
- `scripts/migrate_enrollment_to_vault.py`

---

## 9. What NOT to do (explicitly decided)

| Decision | Reason |
|---|---|  
| ❌ Do NOT delete screenshots | User requirement: kept for PDF gallery |
| ❌ Do NOT add retention / TTL | Not needed right now — storage is manageable |
| ❌ Do NOT use AES immediately | XOR is sufficient for the threat model; AES is the upgrade path |
| ❌ Do NOT change LanceDB vector tables | Already opaque Arrow binary — leave alone |
| ❌ Do NOT change FastAPI endpoint URLs | Frontend compatibility — same routes, different internals |

---

## 10. My Honest Recommendation

If I were making this decision: **implement Phase 1 + Phase 2 this week.**

The portrait store is the highest-risk component — it's the one that directly reveals
identity (filename = entity ID, JPEG bytes = face image). After Phase 2:
- `data/portraits/PRC001.jpg` no longer exists
- `data/vault/portraits/3f/3fa2c8....bin` exists but means nothing to a browser
- Without the `TRACE_VAULT_KEY` in `.env`, the bytes look like noise

Phase 3 (screenshots into vault) is the **next most important** because screenshots
are the largest volume of face data. Phase 4 (enrollment) matters less because
enrollment images are only written once per person and only used at training time.

The gallery algorithm enhancement in Phase 3 is a 2-line change that significantly
improves the PDF output quality — it should be bundled in at minimal cost.

> [!TIP]
> The `data/` folder should also have OS-level ACL set to deny read to all accounts
> except the service account running TRACE-AML. This is the fastest-to-implement
> and most impactful single action, requires zero code changes, and is the correct
> first line of defence regardless of what vault scheme is implemented.

## 1. Current State Audit

### 1.1 What is actually on disk

| Directory | Contents | Issues |
|---|---|---|
| `data/portraits/` | `{entity_id}.jpg` + `{entity_id}.meta.json` | Filenames directly reveal entity IDs. Anyone with filesystem access can map face → ID. |
| `data/screenshots/` | `DET-{timestamp}-{hash}.jpg` (46+ files) | Full-resolution detection frames stored as browsable JPEGs, ~90–120 KB each. No retention policy — grows forever. |
| `data/person_images/{pid}/` | `upload_001.jpg … upload_006.jpg` | Enrollment photos stored in plain readable folders named by person ID. |
| `data/vectors/` | 12× LanceDB `.lance` arrow tables | Embeddings (512-dim float32) are inside binary arrow files — actually the best-protected of all assets already. |
| `data/exports/` | PDF incident reports | Readable PDFs, may include portrait crops and personal details. |
| `data/logs/` | `trace_aml.log` | Logs reference filesystem paths, entity IDs, confidence scores. |
| *(project root)* | `test_photo.jpg` | Test asset accidentally living at root — should not exist in production. |

### 1.2 Current algorithms (what is already good)

| Component | What it does | Quality |
|---|---|---|
| `PortraitStore` | Score-gated updates, 256×256 JPEG-92, per-entity threading lock | ✅ Well-designed |
| `BestCaptureManager` | Confidence hysteresis (5%), keeps only 3 most-recent | ✅ Good |
| `_save_screenshot_crop` | Face-context crop, 320px max, JPEG-80, 5-second throttle per entity | ✅ Already optimised |
| LanceDB vectors | Binary Arrow format, not directly readable as images | ✅ Opaque by nature |

### 1.3 Core problems

```
❌ Problem 1 — Name leakage
   data/portraits/PRC001.jpg  ← entity ID is the filename
   Anyone with file-browser access sees who is in the system.

❌ Problem 2 — Screenshot accumulation
   46 files already, no automatic cleanup, grow unbounded.
   Each is ~90–120 KB JPEG stored at a stable, predictable path.

❌ Problem 3 — Enrollment photos are fully browsable
   data/person_images/PRC001/upload_001.jpg
   Plain JPEG, plain folder name = full dossier visible to OS.

❌ Problem 4 — Absolute paths in database records
   DetectionEvent.screenshot_path stores absolute OS path strings.
   If data is ever moved/copied, all path references break.

❌ Problem 5 — No retention / TTL policy
   Screenshots and detection events are kept forever.
   After weeks of 24/7 operation this will fill a drive.

❌ Problem 6 — Export PDFs are unprotected
   PDF reports include portrait crops and full identity data,
   stored in a plain directory with no access control.
```

---

## 2. Professional Redesign Proposal

### 2.1 Guiding Principles

1. **Opacity by default** — sensitive face images should not have human-readable filenames in the OS file browser.
2. **Content-addressed storage** — file naming derives from a hash of content, not from entity identity.
3. **Relative paths in the database** — never store absolute OS paths; store logical keys that the `DataVault` resolves.
4. **Automatic retention** — every storage bucket has a configurable TTL with an async janitor.
5. **Layered separation** — `portraits` (best shot, kept long-term), `evidence` (screenshots, time-limited), `enrollment` (source images, kept until person is deleted).
6. **Optional lightweight obfuscation** — XOR-scrambling of JPEG bytes prevents casual viewing without requiring full encryption overhead (production deployments can upgrade to AES-128 if needed).

### 2.2 Proposed Directory Layout

```
data/
  vault/
    portraits/        ← {sha256[:2]}/{sha256}.bin  (opaque blobs)
    evidence/         ← {date}/{sha256[:2]}/{sha256}.bin
    enrollment/       ← {pid_hash[:2]}/{pid_hash}.bin
  index/
    portrait_index.json   ← { entity_id → sha256 }  (in-memory on startup)
    evidence_index.db     ← SQLite WAL, maps detection_id → blob key + TTL
  vectors/            ← (unchanged — already opaque LanceDB format)
  exports/            ← (unchanged for now — covered by OS permissions)
  logs/               ← (unchanged)
```

> **Key insight**: The `index/` layer means the OS file browser shows only hashes — zero identity information. The mapping `entity_id → sha256` lives in a small in-memory index file, loaded once at startup.

### 2.3 `DataVault` Abstraction Layer

A new module `src/trace_aml/store/data_vault.py` wraps all image I/O:

```python
class DataVault:
    """
    Content-addressed, opaque binary store for sensitive face assets.
    
    All callers interact with logical keys (entity_id, detection_id),
    never with filesystem paths. The vault resolves keys → file locations
    internally and optionally applies lightweight XOR obfuscation.
    """
    
    def store_portrait(self, entity_id: str, image_bgr: np.ndarray, score: float) -> str:
        """Returns a logical key (sha256). Never returns a filesystem path."""
    
    def get_portrait(self, entity_id: str) -> np.ndarray | None:
        """Returns decoded image array, or None. No path exposed."""
    
    def store_evidence(self, detection_id: str, image_bgr: np.ndarray, ttl_days: int = 30) -> str:
        """Store screenshot evidence. Auto-expires after ttl_days."""
    
    def get_evidence(self, detection_id: str) -> np.ndarray | None:
        """Returns decoded image for this detection, or None if expired/missing."""
    
    def store_enrollment_image(self, person_id: str, image_bgr: np.ndarray) -> str:
        """Store an enrollment/training image for a person."""
    
    def delete_entity(self, entity_id: str) -> None:
        """Remove portrait + all evidence for this entity."""
    
    def run_janitor(self) -> int:
        """Delete expired evidence blobs. Called by background scheduler."""
```

### 2.4 Portrait Endpoint — API stays identical

The FastAPI endpoint `/api/v1/entities/{entity_id}/portrait` stays exactly the same for the frontend. Internally:

```
Before:  FileResponse(portrait_store.get_portrait_path(entity_id))
After:   image_bytes = vault.get_portrait(entity_id) → StreamingResponse(BytesIO(jpeg_bytes))
```

No URL changes, no frontend changes needed.

### 2.5 Screenshot / Evidence Store

**Current behaviour** (already partially good):
- 5-second per-entity throttle ✅
- Face-context crop, 320px max, JPEG-80 ✅
- But: files named `DET-{timestamp}-{hash}.jpg`, browsable, no TTL ❌

**New behaviour**:
- Same throttle and crop algorithm (keep it)
- Store as `data/vault/evidence/{date}/{sha256[:2]}/{sha256}.bin`
- Register in `evidence_index.db` with `expires_at = now + 30 days`
- `DetectionEvent.screenshot_path` stores logical key `sha256` not an OS path
- A background janitor (runs daily or on startup) deletes expired blobs

### 2.6 Enrollment Images (person_images)

**Current**: `data/person_images/PRC001/upload_001.jpg` — fully browsable

**New**: 
- On upload: store via `vault.store_enrollment_image(person_id, img_bgr)`
- The vault computes `sha256(person_id + img_bytes)` → `data/vault/enrollment/{h[:2]}/{h}.bin`
- `enrollment_index.json` maps `{person_id: [sha256_list]}` 
- The training pipeline reads images via vault API — no direct path access

### 2.7 Lightweight Obfuscation (XOR-scrambling)

For operators who want images to be non-openable without running the system:

```python
VAULT_KEY = bytes.fromhex(os.environ.get("TRACE_VAULT_KEY", "00" * 32))

def _scramble(data: bytes) -> bytes:
    """XOR each byte with cycling key. Fast, reversible, stops casual viewing."""
    if all(b == 0 for b in VAULT_KEY):
        return data  # key not set → no obfuscation (dev mode)
    key = VAULT_KEY
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
```

- Set `TRACE_VAULT_KEY` environment variable (64 hex chars = 32 bytes)
- Without the key, `.bin` files show corrupted bytes, not recognisable face images
- With the key, system decodes transparently on read
- **Upgrade path**: replace XOR with `cryptography.fernet.Fernet` for AES-128-CBC if full encryption is required (just swap `_scramble`/`_unscramble`)

### 2.8 Retention Policy Configuration (new config section)

```yaml
# config/config.demo.yaml additions
vault:
  evidence_ttl_days: 30          # Auto-delete screenshots after N days
  janitor_interval_hours: 24     # How often the cleanup job runs
  obfuscation_enabled: false     # true = XOR scramble (set TRACE_VAULT_KEY env var)
  enrollment_keep_on_delete: false  # false = wipe enrollment images when person deleted
```

### 2.9 What stays unchanged (and why)

| Component | Reason to keep |
|---|---|
| `PortraitStore` scoring / hysteresis logic | Already excellent — reuse as the update-decision layer |
| `BestCaptureManager` confidence gating | Same — reuse as the "should I store?" decision layer |
| LanceDB vector tables | Already opaque binary format |
| FastAPI endpoint URLs | Frontend compatibility |
| JPEG compression settings | Already tuned (JPEG-80/92) |

---

## 3. Implementation Plan (Phased)

### Phase 1 — Quick Wins (1–2 days, no breaking changes)
- [ ] Add `VaultSettings` to `config.py` (evidence_ttl_days, janitor_interval_hours)
- [ ] Add a startup janitor in `_lifespan` that scans `data/screenshots/` and deletes files older than TTL
- [ ] Move `test_photo.jpg` from project root into `data/` or delete it
- [ ] Store `screenshot_path` as a relative path (relative to `store.root`) instead of absolute OS path

### Phase 2 — Opaque Storage (2–3 days, backward-compatible migration)
- [ ] Implement `DataVault` in `src/trace_aml/store/data_vault.py`
- [ ] Add `portrait_index.json` — loaded at startup, maps `entity_id → sha256`
- [ ] Migrate `PortraitStore.try_update_portrait` to write via `DataVault`
- [ ] Update portrait GET/POST endpoints to stream from vault (not `FileResponse`)
- [ ] Run migration script: hash-rename existing `data/portraits/*.jpg` into vault layout
- [ ] Update `_save_screenshot_crop` to write into vault evidence store

### Phase 3 — Enrollment Image Vault (1–2 days)
- [ ] Migrate `person_images/` into `vault/enrollment/`
- [ ] Update `person_image_dir()` to use vault-managed paths
- [ ] Update `collect.py` and `train.py` to read via vault API

### Phase 4 — Optional Encryption (when required)
- [ ] Add `TRACE_VAULT_KEY` env var support + XOR scramble
- [ ] Document upgrade path to Fernet AES-128 for classified deployments

---

## 4. Immediate Actions (no code changes needed today)

1. **Add `data/` to `.gitignore`** (confirm it's fully excluded) — check that `portraits/`, `screenshots/`, `person_images/` are all covered
2. **Move `test_photo.jpg`** from project root into `data/` or delete it
3. **Document the data folder** in README so new operators know these dirs contain sensitive biometric data and should be access-controlled at the OS level

> [!IMPORTANT]
> The single most impactful immediate step is **OS-level folder permissions**: restrict `data/` to the service account only. This stops casual browsing without any code changes and is the correct first layer of defence even before the vault refactor.

> [!NOTE]
> The LanceDB vector tables (`data/vectors/`) are already opaque binary (Apache Arrow format). Embeddings inside them are 512-dimensional float32 arrays — mathematically impossible to reconstruct a face image from an embedding alone. This subsystem needs no changes.

> [!TIP]
> The `data/exports/` PDFs are the highest-risk asset since they aggregate identity, face images, and alert history in a human-readable format. These should be the first to get folder-level ACL protection, and eventually PDF password protection could be added to the `PdfReportSettings`.
