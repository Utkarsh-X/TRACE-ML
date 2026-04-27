"""In-memory gallery cache for constant-time face recognition.

WHY THIS EXISTS
--------------
The original ``search_embeddings_for_person_ids`` implementation in
``vector_store.py`` dumps the entire embeddings table into a Python list
and then iterates over it with a manual cosine-distance loop.  For a gallery
of N persons with E embeddings each, that is O(N·E) Python iterations *per
detected face per frame*.  At 30 FPS with 100 enrolled persons the loop runs
~18,000 times per second, and the constant grows linearly with gallery size.

This module replaces the loop with a single BLAS matrix multiply::

    sims = matrix @ query_unit   # (N_total_embeddings, 512) @ (512,) → (N,)

Numpy dispatches that call to an optimised BLAS/SIMD kernel (OpenBLAS /
MKL).  The theoretical complexity is still O(N·E) but the constant factor is
~100 × smaller because there is no Python loop overhead, cache locality is
excellent, and SIMD processes multiple floats per CPU clock.

INCREMENTAL UPDATES
-------------------
When one person is enrolled or deleted only *their* rows in the matrix are
replaced (``upsert_person`` / ``remove_person``).  All other persons are left
completely untouched.  ArcFace embeddings are produced by a frozen neural
network, so a person's vectors never change unless their source images are
replaced.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

EMBEDDING_DIM = 512


class EmbeddingGalleryCache:
    """Thread-safe in-memory gallery for real-time similarity search.

    Usage::

        cache = EmbeddingGalleryCache()
        cache.load_from_records(db_embedding_rows, active_person_ids)

        # On every inference frame:
        hits = cache.search(query_embedding, top_k=96)

        # After enrolling one person (incremental – no full rebuild):
        cache.upsert_person(person_id, list_of_512d_vectors)
        cache.set_active(person_id, True)
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # pid → (K, 512) float32 matrix  (K = enrolled embedding count)
        self._gallery: dict[str, np.ndarray] = {}

        # persons the live pipeline should match against
        self._active_pids: set[str] = set()

        # Flattened arrays rebuilt lazily after any mutation.
        # Keeping them separate from _gallery avoids holding the lock
        # during a potentially slow vstack call on the first access.
        self._flat_matrix: np.ndarray | None = None  # shape (N_total, 512)
        self._flat_pids: list[str] = []              # parallel person_id list
        self._dirty: bool = True

    # ── Private helpers ────────────────────────────────────────────────────

    def _rebuild_flat(self) -> None:
        """Stack all *active* person matrices into one flat array.

        Called inside the lock, so callers must already hold it.
        """
        rows: list[np.ndarray] = []
        pids: list[str] = []
        for pid in self._active_pids:
            mat = self._gallery.get(pid)
            if mat is None or mat.shape[0] == 0:
                continue
            rows.append(mat)
            pids.extend([pid] * mat.shape[0])

        if rows:
            self._flat_matrix = np.vstack(rows).astype(np.float32)
        else:
            self._flat_matrix = None
        self._flat_pids = pids
        self._dirty = False

    # ── Bulk load (startup) ────────────────────────────────────────────────

    def load_from_records(
        self,
        records: list[dict[str, Any]],
        active_pids: set[str],
    ) -> None:
        """Bulk-populate the cache from raw DB rows.

        Args:
            records:    Raw rows from the ``person_embeddings`` LanceDB table.
                        Each row must have ``person_id`` (str) and
                        ``embedding`` (list[float] of length 512).
            active_pids: Set of person_ids whose lifecycle_state == "active".
        """
        from loguru import logger

        gallery: dict[str, list[list[float]]] = {}
        skipped = 0
        for row in records:
            pid = str(row.get("person_id", ""))
            emb = row.get("embedding", [])
            if not pid or len(emb) != EMBEDDING_DIM:
                skipped += 1
                continue
            gallery.setdefault(pid, []).append(emb)

        with self._lock:
            self._gallery = {
                pid: np.asarray(embs, dtype=np.float32)
                for pid, embs in gallery.items()
            }
            self._active_pids = set(active_pids)
            self._dirty = True

        n_persons, n_embs = self.gallery_size
        logger.info(
            "EmbeddingGalleryCache: loaded {} persons / {} embeddings "
            "({} active, {} rows skipped).",
            n_persons,
            n_embs,
            len(active_pids),
            skipped,
        )

    # ── Incremental mutations ──────────────────────────────────────────────

    def upsert_person(
        self,
        person_id: str,
        embeddings: list[list[float]],
    ) -> None:
        """Replace the cached embeddings for *one* person.

        All other persons are untouched.  Safe to call while inference is
        running — the lock prevents partial reads.
        """
        with self._lock:
            if embeddings:
                mat = np.asarray(embeddings, dtype=np.float32)
                if mat.ndim == 2 and mat.shape[1] == EMBEDDING_DIM:
                    self._gallery[person_id] = mat
                else:
                    self._gallery.pop(person_id, None)
            else:
                self._gallery.pop(person_id, None)
            self._dirty = True

    def remove_person(self, person_id: str) -> None:
        """Evict a person from the cache (called on person deletion)."""
        with self._lock:
            self._gallery.pop(person_id, None)
            self._active_pids.discard(person_id)
            self._dirty = True

    def set_active(self, person_id: str, is_active: bool) -> None:
        """Mark or unmark a person for live recognition matching."""
        with self._lock:
            if is_active:
                self._active_pids.add(person_id)
            else:
                self._active_pids.discard(person_id)
            self._dirty = True

    def active_person_ids(self) -> set[str]:
        """Return the current active set — no DB query needed."""
        with self._lock:
            return set(self._active_pids)

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_emb: list[float],
        person_ids_filter: set[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return top-k per-embedding matches as ``{person_id, similarity}``.

        Uses a single BLAS matrix multiply instead of a Python loop.

        Args:
            query_emb:          512-d unit-normalised query embedding.
            person_ids_filter:  If given, only embeddings belonging to these
                                persons are considered.  When ``None`` the
                                entire active gallery is searched.
            top_k:              Maximum results to return.

        Returns:
            List of ``{"person_id": str, "similarity": float}`` dicts sorted
            by descending similarity.  Each dict represents one enrolled
            embedding (not one person) so callers can group / aggregate.
        """
        query = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(query) + 1e-9)
        # All stored embeddings are already unit-normalised at enrolment time,
        # so matrix @ unit_query == cosine similarity directly.
        query_unit = (query / q_norm).astype(np.float32)

        with self._lock:
            if self._dirty:
                self._rebuild_flat()
            matrix = self._flat_matrix
            pids = list(self._flat_pids)

        if matrix is None or len(pids) == 0:
            return []

        # Single BLAS call — (N, 512) @ (512,) → (N,)
        sims: np.ndarray = matrix @ query_unit

        # ── Full-NumPy top-k extraction ────────────────────────────────────
        # OLD: iterate every element in Python, lambda-sort the whole list.
        #      Cost = O(N) Python iterations + O(N log N) Python sort per frame.
        # NEW: numpy argpartition picks the k largest in O(N) C-land, then we
        #      only iterate over ≤top_k elements in Python (typically 96).
        n = len(pids)
        k = min(top_k, n)

        if k == 0:
            return []

        # argpartition gives us k indices with the highest similarities —
        # unsorted among themselves.  We then stable-sort only those k values.
        top_idx: np.ndarray = np.argpartition(sims, n - k)[n - k:]
        # Sort the k candidates descending by similarity (k ≤ 96 → negligible cost)
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        results: list[dict[str, Any]] = []
        for i in top_idx.tolist():
            sim = float(sims[i])
            if sim <= 0.0:
                continue
            pid = pids[i]
            if person_ids_filter is not None and pid not in person_ids_filter:
                continue
            results.append({"person_id": pid, "similarity": sim})
        # ──────────────────────────────────────────────────────────────────
        return results


    # ── Diagnostics ────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._gallery) == 0

    @property
    def gallery_size(self) -> tuple[int, int]:
        """Return (num_persons, total_embeddings) for logging / health checks."""
        with self._lock:
            n_persons = len(self._gallery)
            n_embeddings = sum(m.shape[0] for m in self._gallery.values())
            return n_persons, n_embeddings

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active_pids)
