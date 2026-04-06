import numpy as np

from trace_aml.core.config import Settings
from trace_aml.core.models import FaceCandidate
from trace_aml.recognizers.arcface import ArcFaceRecognizer


class _RecognizerStub(ArcFaceRecognizer):
    def detect_faces(self, frame_bgr: np.ndarray):
        emb = [0.0] * 512
        emb[0] = 1.0
        return [FaceCandidate(bbox=(10, 10, 100, 100), embedding=emb, detector_score=0.9)]


class _StoreStub:
    def active_person_ids(self):
        return {"PRC004"}

    def search_embeddings_for_person_ids(self, embedding, person_ids, top_k=5):
        if "PRC004" in set(person_ids):
            return [{"person_id": "PRC004", "similarity": 0.89}]
        return []

    def search_embeddings(self, embedding, top_k=5):
        return [
            {"person_id": "PRC005", "similarity": 0.93},
            {"person_id": "PRC004", "similarity": 0.89},
        ]

    def get_person(self, person_id: str):
        if person_id == "PRC004":
            return {"person_id": "PRC004", "name": "arjun_v2", "category": "criminal", "lifecycle_state": "active"}
        if person_id == "PRC005":
            return {"person_id": "PRC005", "name": "arjun_v2 dup", "category": "criminal", "lifecycle_state": "ready"}
        return None


def test_match_prefers_active_candidate_over_higher_non_active_similarity() -> None:
    settings = Settings()
    recognizer = _RecognizerStub(settings)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    outputs = recognizer.match(frame, _StoreStub())
    assert len(outputs) == 1
    match = outputs[0][0]
    assert match.person_id == "PRC004"
    assert match.name == "arjun_v2"
    assert "person_not_active" not in match.quality_flags
