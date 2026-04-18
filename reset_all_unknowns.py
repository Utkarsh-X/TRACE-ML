"""
Clear all accumulated UNK entities, profiles, events and portraits — fresh start.
PRM001 (enrolled person) is preserved.
"""
from trace_aml.core.config import load_settings
from trace_aml.store.vector_store import VectorStore
from pathlib import Path

cfg = load_settings("config/config.demo.yaml")
store = VectorStore(cfg)

# Find all UNK entity IDs
all_entities = store._query_rows(store.entities, limit=10000)
unk_ids = [e.get("entity_id") for e in all_entities if str(e.get("entity_id", "")).startswith("UNK")]
print("UNK entities to clear:", unk_ids)

for eid in unk_ids:
    escaped = store._escape(eid)
    store.unknown_profiles.delete(f"entity_id = '{escaped}'")
    store.entities.delete(f"entity_id = '{escaped}'")
    store.events.delete(f"entity_id = '{escaped}'")
    store.alerts.delete(f"entity_id = '{escaped}'")
    store.incidents.delete(f"entity_id = '{escaped}'")

# Also clear UNK portraits left on disk
portraits_dir = Path(cfg.store.portraits_dir)
removed = []
for f in portraits_dir.glob("UNK*"):
    f.unlink()
    removed.append(f.name)

print(f"Cleared {len(unk_ids)} entities, {len(removed)} portrait files")
print("Remaining entities:", [e.get("entity_id") for e in store._query_rows(store.entities, limit=100)])
print("Remaining portraits:", [f.name for f in portraits_dir.iterdir()])
