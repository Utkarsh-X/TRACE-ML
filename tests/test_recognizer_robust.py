from trace_aml.core.config import Settings
from trace_aml.recognizers.arcface import ArcFaceRecognizer


class _StoreStub:
    def get_person(self, person_id: str):
        return {
            "person_id": person_id,
            "name": person_id,
            "category": "criminal",
            "lifecycle_state": "active" if person_id == "P1" else "ready",
        }


def test_effective_thresholds_relax_for_low_quality() -> None:
    recognizer = ArcFaceRecognizer(Settings())
    hi_accept, hi_review = recognizer._effective_thresholds(face_quality=0.95)
    lo_accept, lo_review = recognizer._effective_thresholds(face_quality=0.10)
    assert lo_accept < hi_accept
    assert lo_review < hi_review


def test_aggregate_person_scores_rewards_stable_support() -> None:
    recognizer = ArcFaceRecognizer(Settings())
    rows = [
        {"person_id": "P1", "similarity": 0.68},
        {"person_id": "P1", "similarity": 0.67},
        {"person_id": "P1", "similarity": 0.66},
        {"person_id": "P2", "similarity": 0.70},
    ]
    ranked = recognizer._aggregate_person_scores(rows, _StoreStub(), active_ids={"P1"})
    assert ranked
    assert ranked[0]["person_id"] == "P1"
    assert ranked[0]["is_active"] is True
    assert ranked[0]["support"] == 3
