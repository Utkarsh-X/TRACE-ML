from trace_aml.core.config import load_settings
from trace_aml.store.vector_store import VectorStore
from collections import Counter

cfg = load_settings("config/config.demo.yaml")
store = VectorStore(cfg)

print("=== PERSONS ===")
persons = store._query_rows(store.persons)
for p in persons:
    print("  id={} name={} lifecycle={} category={}".format(
        p.get("person_id"), p.get("name"), p.get("lifecycle_state"), p.get("category")))

print()
print("=== EMBEDDINGS COUNT PER PERSON ===")
embs = store._query_rows(store.embeddings)
counts = Counter(str(e.get("person_id")) for e in embs)
for pid, cnt in sorted(counts.items()):
    print("  {}: {} embeddings".format(pid, cnt))

print()
print("=== ACTIVE PERSON IDS (searched by recognizer) ===")
active = store.active_person_ids()
print(" ", active)

print()
print("=== CONFIG THRESHOLDS ===")
print("  min_embeddings_active =", cfg.quality.min_embeddings_active)
print("  min_embeddings_ready  =", cfg.quality.min_embeddings_ready)
print("  min_quality_score     =", cfg.quality.min_quality_score)
print("  accept_threshold      =", cfg.recognition.accept_threshold)
print("  review_threshold      =", cfg.recognition.review_threshold)
