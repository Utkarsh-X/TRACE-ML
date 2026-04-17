"""Background unknown-entity graph clusterer.

Architecture
------------
Every ``interval_minutes`` this daemon runs a *global* pairwise embedding
comparison across all stored unknown entity profiles.  It builds a similarity
graph where an edge (A, B) exists if **any** embedding from A and **any**
embedding from B score >= ``merge_threshold`` against each other (best-of-N
comparison).  Connected components of this graph correspond to the same real
person seen under different conditions.

The oldest UNK entity ID in each component is kept as the *primary*.
All events, alerts, incidents and embeddings from the other members are
re-pointed to it and their entity records deleted.  If the primary has no
portrait, the best one from the duplicates is copied over.

An SSE ``entity_merge`` event is broadcast so the frontend refreshes
automatically without polling.

Compute cost
------------
For 50 unknown entities × 8 embeddings each: one 400×400 float32 matmul
(~640 KB) takes < 2 ms in numpy on CPU.  The subsequent per-entity-pair max
extraction is an O(E² K²) Python loop but with tiny numbers in practice.
Total wall-clock cost at 100 unknowns: < 50 ms.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from trace_aml.core.config import Settings
    from trace_aml.core.streaming import EventStreamPublisher
    from trace_aml.store.vector_store import VectorStore

EMBEDDING_DIM = 512


# ── Union-Find ───────────────────────────────────────────────────────────────

class _UnionFind:
    """Lightweight path-compressed union-find for connected-component detection."""

    def __init__(self, items: list[str]) -> None:
        self._parent: dict[str, str] = {x: x for x in items}

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Keep lexicographically smaller ID as root → preserves oldest UNK ID
        if ra < rb:
            self._parent[rb] = ra
        else:
            self._parent[ra] = rb

    def components(self) -> dict[str, list[str]]:
        """Return ``{root: [sorted members]}`` for every component with ≥ 2 members."""
        groups: dict[str, list[str]] = {}
        for item in self._parent:
            root = self.find(item)
            groups.setdefault(root, []).append(item)
        return {r: sorted(ms) for r, ms in groups.items() if len(ms) > 1}


# ── Main Clusterer ───────────────────────────────────────────────────────────

class UnknownEntityClusterer:
    """Daemon thread that retroactively merges duplicate unknown entities.

    Usage::

        clusterer = UnknownEntityClusterer(settings, store, publisher)
        clusterer.start()   # call when recognition is enabled
        ...
        clusterer.stop()    # call when recognition is disabled
        merges = clusterer.run_once()  # manual trigger (blocking)
    """

    def __init__(
        self,
        settings: "Settings",
        store: "VectorStore",
        publisher: "EventStreamPublisher | None" = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._publisher = publisher
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background daemon thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        cfg = self._settings.unknown_clustering
        if not cfg.enabled:
            logger.info("CLUSTER: background clusterer disabled by config")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="unk-entity-clusterer",
        )
        self._thread.start()
        logger.info(
            f"CLUSTER: clusterer started  "
            f"interval={cfg.interval_minutes}m  "
            f"threshold={cfg.merge_threshold}"
        )

    def stop(self) -> None:
        """Signal the daemon to stop and wait (up to 5 s)."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        logger.info("CLUSTER: clusterer stopped")

    def run_once(self) -> int:
        """Perform one clustering pass synchronously.  Returns merge count."""
        return self._cluster_unknowns()

    # ── Internal loop ────────────────────────────────────────────────────────

    def _loop(self) -> None:
        cfg = self._settings.unknown_clustering
        interval_sec = max(10.0, cfg.interval_minutes * 60.0)

        # Wait a short warm-up period before the first pass
        warmup = min(30.0, interval_sec * 0.5)
        if self._stop_event.wait(warmup):
            return

        while not self._stop_event.is_set():
            try:
                merges = self._cluster_unknowns()
                if merges > 0:
                    logger.info(f"CLUSTER: pass complete — {merges} merge(s)")
            except Exception as exc:
                logger.warning(f"CLUSTER: pass error: {exc}")

            # Sleep in 1-second ticks so stop() is responsive
            deadline = time.monotonic() + interval_sec
            while time.monotonic() < deadline:
                if self._stop_event.wait(1.0):
                    return

    # ── Core algorithm ───────────────────────────────────────────────────────

    def _cluster_unknowns(self) -> int:
        """One full graph-clustering pass.  Returns number of merges done."""
        cfg = self._settings.unknown_clustering

        # ── 1. Load ALL unknown-profile rows from DB ──────────────────────────
        all_rows = self._store._query_rows(
            self._store.unknown_profiles, limit=200_000
        )
        if not all_rows:
            return 0

        # Group normalised embeddings by entity_id
        entity_embs: dict[str, list[np.ndarray]] = {}
        for row in all_rows:
            eid = str(row.get("entity_id", "")).strip()
            if not eid:
                continue
            raw = row.get("embedding", [])
            vec = np.asarray(raw, dtype=np.float32).reshape(-1)
            if vec.shape[0] != EMBEDDING_DIM:
                continue
            norm = float(np.linalg.norm(vec)) + 1e-9
            entity_embs.setdefault(eid, []).append(
                (vec / norm).astype(np.float32)
            )

        # Skip entities with too few embeddings to trust yet
        min_embs = max(1, cfg.min_embeddings_to_cluster)
        entity_embs = {k: v for k, v in entity_embs.items() if len(v) >= min_embs}

        if len(entity_embs) < 2:
            return 0

        # ── 2. Stack all embeddings into one matrix (one BLAS call) ──────────
        entity_list = sorted(entity_embs.keys())  # stable: older IDs first
        rows_per_entity: dict[str, list[int]] = {}
        all_vecs: list[np.ndarray] = []
        for eid in entity_list:
            start = len(all_vecs)
            all_vecs.extend(entity_embs[eid])
            rows_per_entity[eid] = list(range(start, len(all_vecs)))

        emb_matrix = np.stack(all_vecs, axis=0)  # (N, 512) float32
        # Full pairwise cosine similarity matrix in a single BLAS matmul
        sim_matrix: np.ndarray = emb_matrix @ emb_matrix.T  # (N, N)

        # ── 3. Build similarity graph with union-find ─────────────────────────
        uf = _UnionFind(entity_list)
        threshold = cfg.merge_threshold

        for i, ei in enumerate(entity_list):
            ri = rows_per_entity[ei]
            for j in range(i + 1, len(entity_list)):
                ej = entity_list[j]
                rj = rows_per_entity[ej]
                # Best-of-N×M cosine similarity between entity i and entity j
                sub = sim_matrix[np.ix_(ri, rj)]
                if sub.size > 0 and float(sub.max()) >= threshold:
                    uf.union(ei, ej)

        # ── 4. Merge each multi-member component ──────────────────────────────
        components = uf.components()
        if not components:
            return 0

        total = 0
        for primary_id, members in components.items():
            duplicates = [m for m in members if m != primary_id]
            if not duplicates:
                continue

            logger.debug(
                f"CLUSTER: component {primary_id} ← {duplicates}"
            )

            # Copy best portrait to primary before DB merge
            self._merge_portraits(primary_id, duplicates)

            # Atomically re-point all DB records and remove duplicates
            n = self._store.merge_entities(
                primary_id=primary_id,
                duplicate_ids=duplicates,
            )
            total += n

            # Broadcast SSE so the frontend refreshes immediately
            if n > 0 and self._publisher is not None:
                try:
                    self._publisher.publish(
                        "entity_merge",
                        {"primary": primary_id, "merged": duplicates, "count": n},
                    )
                except Exception:
                    pass  # publisher errors must never crash the daemon

        return total

    # ── Portrait handling ────────────────────────────────────────────────────

    def _merge_portraits(
        self,
        primary_id: str,
        duplicate_ids: list[str],
    ) -> None:
        """Copy the best available portrait from duplicates to primary (if needed)."""
        portraits_dir = Path(self._settings.store.portraits_dir)
        primary_jpg = portraits_dir / f"{primary_id}.jpg"

        if primary_jpg.exists():
            return  # primary already has a good portrait — leave it alone

        for dup_id in duplicate_ids:
            dup_jpg = portraits_dir / f"{dup_id}.jpg"
            dup_meta = portraits_dir / f"{dup_id}.meta.json"
            if not dup_jpg.exists():
                continue
            try:
                shutil.copy2(str(dup_jpg), str(primary_jpg))
                # Copy meta sidecar so the portrait score is preserved
                meta_dest = portraits_dir / f"{primary_id}.meta.json"
                if dup_meta.exists() and not meta_dest.exists():
                    shutil.copy2(str(dup_meta), str(meta_dest))
                logger.debug(f"CLUSTER: portrait {dup_id}.jpg → {primary_id}.jpg")
                break  # one portrait is enough
            except Exception as exc:
                logger.debug(f"CLUSTER: portrait copy error: {exc}")
