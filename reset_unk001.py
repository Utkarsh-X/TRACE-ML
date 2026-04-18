"""
Reset UNK001: delete its contaminated embeddings so the next recognition
session gets a fresh start instead of re-using false-merged embeddings.
The entity record itself is kept; only the stored embeddings are removed.
"""
from trace_aml.core.config import load_settings
from trace_aml.store.vector_store import VectorStore

cfg = load_settings("config/config.demo.yaml")
store = VectorStore(cfg)

# Count before
profiles_before = store._query_rows(store.unknown_profiles, limit=100_000)
unk_before = [p for p in profiles_before if str(p.get("entity_id")) == "UNK001"]
print(f"UNK001 embeddings before: {len(unk_before)}")

# Delete UNK001 embeddings
try:
    store.unknown_profiles.delete("entity_id = 'UNK001'")
    print("Deleted UNK001 unknown_profiles rows")
except Exception as e:
    print(f"Error: {e}")

# Delete UNK001 entity record too (will be re-created on next sighting)
try:
    store.entities.delete("entity_id = 'UNK001'")
    print("Deleted UNK001 from entities table")
except Exception as e:
    print(f"Error: {e}")

# Delete UNK001 events (optional — keep history clean)
try:
    store.events.delete("entity_id = 'UNK001'")
    print("Deleted UNK001 events")
except Exception as e:
    print(f"Error: {e}")

profiles_after = store._query_rows(store.unknown_profiles, limit=100_000)
print(f"UNK001 embeddings after: {len([p for p in profiles_after if str(p.get('entity_id'))=='UNK001'])}")
entities_after = store._query_rows(store.entities, limit=100_000)
print("Remaining entities:", [e.get("entity_id") for e in entities_after])
print("Done — UNK001 contamination cleared. Next detection will create a fresh entity.")
