"""
Quick diagnostic: check unknown_profiles and entities tables.
Run: .venv\Scripts\python.exe diag2.py
"""
from trace_aml.core.config import load_settings
from trace_aml.store.vector_store import VectorStore

cfg = load_settings("config/config.demo.yaml")
store = VectorStore(cfg)

print("=== UNKNOWN_PROFILES (entity_id, sample_count, quality) ===")
profiles = store._query_rows(store.unknown_profiles, limit=1000)
from collections import Counter
eid_counts = Counter(str(p.get("entity_id")) for p in profiles)
for eid, cnt in sorted(eid_counts.items()):
    print("  entity_id={} embeddings={}".format(eid, cnt))
print("Total profile rows:", len(profiles))

print()
print("=== ENTITIES TABLE ===")
entities = store._query_rows(store.entities, limit=1000)
for e in entities:
    print("  id={} type={} status={} last_seen={}".format(
        e.get("entity_id"), e.get("entity_type"), e.get("status"), str(e.get("last_seen_at",""))[:19]))

print()
print("=== EVENTS (per entity) ===")
events = store._query_rows(store.events, limit=100000)
event_counts = Counter(str(ev.get("entity_id")) for ev in events)
for eid, cnt in sorted(event_counts.items()):
    print("  entity_id={} events={}".format(eid, cnt))
