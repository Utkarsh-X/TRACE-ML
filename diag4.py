"""
Full system state diagnostic — persons, states, embeddings, entities.
"""
from trace_aml.core.config import load_settings
from trace_aml.store.vector_store import VectorStore

cfg = load_settings("config/config.demo.yaml")
store = VectorStore(cfg)

print("=== PERSONS TABLE ===")
for p in store._query_rows(store.persons, limit=100):
    print("  person_id={} name={} category={}".format(
        p.get("person_id"), p.get("name"), p.get("category")))

print("\n=== PERSON_STATES TABLE ===")
for p in store._query_rows(store.person_states, limit=100):
    print("  person_id={} lifecycle={} embeddings={} score={:.3f}".format(
        p.get("person_id"), p.get("lifecycle_state"),
        p.get("valid_embeddings"), float(p.get("enrollment_score", 0))))

print("\n=== EMBEDDINGS TABLE (grouped) ===")
from collections import Counter
embs = store._query_rows(store.embeddings, limit=100000)
for pid, cnt in Counter(e.get("person_id") for e in embs).items():
    print("  person_id={} count={}".format(pid, cnt))

print("\n=== ENTITIES TABLE (full) ===")
for e in store._query_rows(store.entities, limit=100):
    print("  entity_id={} type={!r} status={} last_seen={}".format(
        e.get("entity_id"), e.get("type"), e.get("status"),
        str(e.get("last_seen_at",""))[:19]))

print("\n=== UNKNOWN_PROFILES (grouped) ===")
up = store._query_rows(store.unknown_profiles, limit=100000)
for eid, cnt in Counter(r.get("entity_id") for r in up).items():
    print("  entity_id={} count={}".format(eid, cnt))

print("\n=== list_entities(type_filter='unknown') ===")
unk = store.list_entities(type_filter="unknown")
print("  results:", [e.get("entity_id") for e in unk])

print("\n=== list_entities(type_filter='known') ===")
kn = store.list_entities(type_filter="known")
print("  results:", [e.get("entity_id") for e in kn])
