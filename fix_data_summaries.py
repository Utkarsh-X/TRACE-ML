from trace_aml.core.config import load_settings
from trace_aml.store.vector_store import VectorStore
from trace_aml.core.models import IncidentRecord, IncidentStatus, AlertSeverity

def fix_data():
    settings = load_settings()
    store = VectorStore(settings)
    incidents = store.list_incidents(limit=10000)
    fixed_count = 0
    
    print(f"Checking {len(incidents)} incidents...")
    
    for inc in incidents:
        summary = str(inc.get("summary", ""))
        if ": " in summary:
            parts = summary.split(": ")
            prefix = parts[0]
            rest = ": ".join(parts[1:]).strip()
            
            if rest.upper().startswith(prefix.upper()):
                # Redundancy found
                new_rest = rest[len(prefix):].strip().lstrip(":").lstrip("-").strip()
                if new_rest:
                    new_rest = new_rest[0].upper() + new_rest[1:]
                else:
                    new_rest = "Detected"
                
                new_summary = f"{prefix}: {new_rest}"
                print(f"Fixing ID {inc['incident_id']}: '{summary}' -> '{new_summary}'")
                
                model = IncidentRecord(
                    incident_id=str(inc["incident_id"]),
                    entity_id=str(inc["entity_id"]),
                    status=IncidentStatus(str(inc["status"]).lower()),
                    start_time=str(inc["start_time"]),
                    last_seen_time=str(inc["last_seen_time"]),
                    alert_ids=inc["alert_ids"],
                    alert_count=int(inc["alert_count"]),
                    severity=AlertSeverity(str(inc["severity"]).lower()),
                    summary=new_summary,
                    last_action_at=str(inc.get("last_action_at", ""))
                )
                store.update_incident(model)
                fixed_count += 1
                
    print(f"Done. Fixed {fixed_count} records.")

if __name__ == "__main__":
    fix_data()
