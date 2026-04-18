"""
Diagnostic: test entity type round-trip and identify root causes.
"""
from trace_aml.core.config import load_settings
from trace_aml.core.models import EntityRecord, EntityType, EntityStatus
from trace_aml.store.vector_store import VectorStore

cfg = load_settings("config/config.demo.yaml")
store = VectorStore(cfg)

# 1. What does model_dump() produce?
payload = EntityRecord(
    entity_id="TEST_TYPE",
    type=EntityType.unknown,
    status=EntityStatus.active,
    source_person_id=None,
    created_at="2026-01-01T00:00:00Z",
    last_seen_at="2026-01-01T00:00:00Z",
).model_dump()
print("1. model_dump():", payload)
print("   type field python type:", type(payload.get("type")))
print("   type field value repr:", repr(payload.get("type")))

# 2. Entity table columns
cols = store._table_columns(store.entities)
print("\n2. Entity table columns:", sorted(cols))

# 3. What _filtered_row produces
filtered = store._filtered_row(store.entities, payload)
print("\n3. _filtered_row():", filtered)

# 4. Write then read back with raw LanceDB API
store.entities.delete("entity_id = 'TEST_TYPE'")
store.entities.add([filtered])

raw = store.entities.search().where("entity_id = 'TEST_TYPE'", prefilter=True).limit(1).to_list()
print("\n4. Raw read from LanceDB:", raw)

# 5. Via _query_rows
qr = store._query_rows(store.entities, where="entity_id = 'TEST_TYPE'", limit=1)
print("\n5. _query_rows():", qr)

# 6. Show what list_entities(type_filter=...) returns
unk = store.list_entities(type_filter=EntityType.unknown.value)
print("\n6. list_entities(type_filter='unknown') count:", len(unk), "->", [e.get('entity_id') for e in unk])

# Cleanup
store.entities.delete("entity_id = 'TEST_TYPE'")

# 7. Show all actual entity rows
print("\n7. ALL entities in table:")
for row in store._query_rows(store.entities, limit=100):
    print("  ", {k: row.get(k) for k in ("entity_id", "type", "status")})
