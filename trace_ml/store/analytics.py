"""DuckDB-backed analytics helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from trace_ml.core.models import HistoryQuery, SummaryReport
from trace_ml.store.vector_store import VectorStore


class AnalyticsStore:
    """Runs analytical SQL over detection snapshots."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self.conn = duckdb.connect()

    def _register_detections(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            arrow_table = pa.table(
                {
                    "detection_id": pa.array([], type=pa.string()),
                    "timestamp_utc": pa.array([], type=pa.string()),
                    "source": pa.array([], type=pa.string()),
                    "person_id": pa.array([], type=pa.string()),
                    "name": pa.array([], type=pa.string()),
                    "category": pa.array([], type=pa.string()),
                    "confidence": pa.array([], type=pa.float32()),
                    "similarity": pa.array([], type=pa.float32()),
                    "smoothed_confidence": pa.array([], type=pa.float32()),
                    "decision_state": pa.array([], type=pa.string()),
                    "decision_reason": pa.array([], type=pa.string()),
                    "track_id": pa.array([], type=pa.string()),
                    "quality_flags": pa.array([], type=pa.string()),
                    "bbox": pa.array([], type=pa.string()),
                    "screenshot_path": pa.array([], type=pa.string()),
                }
            )
        else:
            normalized = []
            for row in rows:
                flags = row.get("quality_flags", [])
                if isinstance(flags, list):
                    flags_value = ",".join([str(v) for v in flags])
                else:
                    flags_value = str(flags)
                normalized.append(
                    {
                        "detection_id": row.get("detection_id", ""),
                        "timestamp_utc": row.get("timestamp_utc", ""),
                        "source": row.get("source", ""),
                        "person_id": row.get("person_id", ""),
                        "name": row.get("name", ""),
                        "category": row.get("category", ""),
                        "confidence": float(row.get("confidence", 0.0)),
                        "similarity": float(row.get("similarity", 0.0)),
                        "smoothed_confidence": float(row.get("smoothed_confidence", row.get("confidence", 0.0))),
                        "decision_state": str(row.get("decision_state", "reject")),
                        "decision_reason": str(row.get("decision_reason", "")),
                        "track_id": str(row.get("track_id", "")),
                        "quality_flags": flags_value,
                        "bbox": str(row.get("bbox", "")),
                        "screenshot_path": row.get("screenshot_path", ""),
                    }
                )
            arrow_table = pa.Table.from_pylist(normalized)
        self.conn.register("detections_view", arrow_table)

    def history(self, query: HistoryQuery) -> list[dict[str, Any]]:
        rows = self.vector_store.list_detections(limit=100_000)
        self._register_detections(rows)

        clauses = ["1=1"]
        params: list[Any] = []
        if query.start_ts:
            clauses.append("timestamp_utc >= ?")
            params.append(query.start_ts)
        if query.end_ts:
            clauses.append("timestamp_utc <= ?")
            params.append(query.end_ts)
        if query.person_id:
            clauses.append("person_id = ?")
            params.append(query.person_id)
        if query.category:
            clauses.append("category = ?")
            params.append(query.category)
        if query.decision_state:
            clauses.append("decision_state = ?")
            params.append(query.decision_state)
        clauses.append("confidence >= ?")
        params.append(float(query.min_confidence))

        sql = f"""
            SELECT detection_id, timestamp_utc, source, person_id, name, category,
                   confidence, similarity, smoothed_confidence, decision_state, decision_reason,
                   track_id, quality_flags, bbox, screenshot_path
            FROM detections_view
            WHERE {" AND ".join(clauses)}
            ORDER BY timestamp_utc DESC
            LIMIT {int(query.limit)}
        """
        result = self.conn.execute(sql, params).fetchall()
        cols = [desc[0] for desc in self.conn.description]
        return [dict(zip(cols, row, strict=False)) for row in result]

    def summary(self) -> SummaryReport:
        rows = self.vector_store.list_detections(limit=100_000)
        self._register_detections(rows)

        agg = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_detections,
                COUNT(DISTINCT NULLIF(person_id, '')) AS unique_persons,
                COALESCE(AVG(confidence), 0.0) AS avg_confidence
            FROM detections_view
            """
        ).fetchone()

        top_rows = self.conn.execute(
            """
            SELECT name, category, COUNT(*) AS hits, ROUND(AVG(confidence), 3) AS avg_conf
            FROM detections_view
            GROUP BY name, category
            ORDER BY hits DESC
            LIMIT 5
            """
        ).fetchall()

        decision_rows = self.conn.execute(
            """
            SELECT decision_state, COUNT(*) AS cnt
            FROM detections_view
            GROUP BY decision_state
            """
        ).fetchall()
        distribution = {str(row[0]): int(row[1]) for row in decision_rows}

        states = self.vector_store.list_person_states()
        blocked = sum(1 for row in states if row.get("lifecycle_state") == "blocked")
        low_quality = sum(
            1
            for row in states
            if float(row.get("enrollment_score", 0.0)) < self.vector_store.settings.quality.min_quality_score
        )

        return SummaryReport(
            total_detections=int(agg[0]),
            unique_persons=int(agg[1]),
            avg_confidence=float(agg[2]),
            decision_distribution=distribution,
            blocked_persons=blocked,
            low_quality_persons=low_quality,
            top_persons=[
                {
                    "name": row[0],
                    "category": row[1],
                    "hits": int(row[2]),
                    "avg_confidence": float(row[3]),
                }
                for row in top_rows
            ],
        )

    def low_quality_enrollments(self, limit: int = 20) -> list[dict[str, Any]]:
        persons = self.vector_store.list_persons()
        rows: list[dict[str, Any]] = []
        threshold = self.vector_store.settings.quality.min_quality_score
        for person in persons:
            score = float(person.get("enrollment_score", 0.0))
            state = str(person.get("lifecycle_state", "draft"))
            if score < threshold or state in {"draft", "blocked"}:
                rows.append(
                    {
                        "person_id": person.get("person_id", ""),
                        "name": person.get("name", ""),
                        "state": state,
                        "enrollment_score": score,
                        "valid_embeddings": int(person.get("valid_embeddings", 0)),
                        "valid_images": int(person.get("valid_images", 0)),
                        "total_images": int(person.get("total_images", 0)),
                        "reason": person.get("lifecycle_reason", ""),
                    }
                )
        rows.sort(key=lambda r: (r["state"] != "blocked", r["enrollment_score"]))
        return rows[:limit]

    def threshold_impact(self) -> dict[str, int]:
        rows = self.vector_store.list_detections(limit=100_000)
        accept_thr = self.vector_store.settings.recognition.accept_threshold * 100.0
        review_thr = self.vector_store.settings.recognition.review_threshold * 100.0
        accepted = 0
        review = 0
        reject = 0
        for row in rows:
            score = float(row.get("smoothed_confidence", row.get("confidence", 0.0)))
            if score >= accept_thr:
                accepted += 1
            elif score >= review_thr:
                review += 1
            else:
                reject += 1
        return {"accept_band": accepted, "review_band": review, "reject_band": reject}

    def export_csv(self, query: HistoryQuery, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = self.history(query)
        fieldnames = [
            "detection_id",
            "timestamp_utc",
            "source",
            "person_id",
            "name",
            "category",
            "confidence",
            "similarity",
            "smoothed_confidence",
            "decision_state",
            "decision_reason",
            "track_id",
            "quality_flags",
            "bbox",
            "screenshot_path",
        ]
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return output
