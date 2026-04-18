"""Fix PRC001: write active state to person_states table, then restart."""
from trace_aml.core.config import load_settings
from trace_aml.store.vector_store import VectorStore

cfg = load_settings("config/config.demo.yaml")
store = VectorStore(cfg)

# --- Diagnose person_states for PRC001 ---
state = store.get_person_state("PRC001")
print("person_states entry:", state)

embs = store._query_rows(store.embeddings,
                         where="person_id = 'PRC001'", limit=100)
print("Embeddings in DB:", len(embs))

# --- Fix: write active state ---
store.set_person_state(
    person_id="PRC001",
    lifecycle_state="active",
    lifecycle_reason="manually_activated_buffalo_l_migration",
    enrollment_score=0.66,
    valid_embeddings=len(embs),
    valid_images=len(embs),
    total_images=len(embs),
)
print("set_person_state -> active")

# --- Verify ---
state2 = store.get_person_state("PRC001")
print("After:", state2)
active = store.active_person_ids()
print("active_person_ids:", active)
print("PRC001 active:", "PRC001" in active)
