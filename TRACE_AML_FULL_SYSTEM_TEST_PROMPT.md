# TRACE-AML Full System Verification Prompt (Pre-Service / Pre-GUI Gate)

Use this document as the single execution script before moving to Service Layer and GUI.

## 0) Ground Rules
- Run in project root.
- Use the isolated env only: `.venv311`.
- Use real IDs in commands. Do **not** type placeholders with `<` or `>` in PowerShell.
- Keep this as a validation gate: if a critical test fails, stop and fix before proceeding.

---

## 1) Environment + Automated Test Gate

```powershell
& ".\.venv311\Scripts\Activate.ps1"
python -V
python -m pytest -q
python -m trace_ml --help
python -m trace_ml person --help
python -m trace_ml events --help
python -m trace_ml alerts --help
python -m trace_ml incident --help
python -m trace_ml action --help
```

Pass criteria:
- `pytest` passes fully.
- CLI command groups are visible, including `alerts`, `incident`, `action`.

---

## 2) Baseline Data Snapshot

```powershell
python -m trace_ml person list
python -m trace_ml person audit
python -m trace_ml report summary
python -m trace_ml report quality
```

Record:
- Active/ready/blocked persons.
- Current low-quality enrollments.

---

## 3) Core Pipeline Validation (Detection -> Event -> Alert -> Incident -> Action)

## Scenario A: Known Entity Stability
1. Start live:
```powershell
python -m trace_ml --config config.demo.yaml recognize live
```
2. Stay in frame continuously for 30-60 seconds, then quit with `q`.
3. Validate:
```powershell
python -m trace_ml history query --limit 30
python -m trace_ml events tail --limit 30
python -m trace_ml alerts tail --limit 30
python -m trace_ml incident list --status open --limit 30
```

Expected:
- Stable recognized entity.
- Events flowing.
- Alerts present only if rule windows justify them.
- No incident explosion (single active incident per entity).

## Scenario B: Unknown Recurrence
1. Run live again and create unknown pattern:
- hide face / occlude / leave frame / return repeatedly.
2. Validate:
```powershell
python -m trace_ml alerts tail --limit 30
python -m trace_ml incident list --status open --limit 30
```
3. Pick real incident ID from list and inspect:
```powershell
python -m trace_ml incident show --id REAL_INCIDENT_ID
```

Expected:
- Unknown recurrence/reappearance style alerts.
- Incident created and accumulating alerts.

## Scenario C: Reappearance Burst
1. Enter/exit frame quickly multiple times in one live session.
2. Validate:
```powershell
python -m trace_ml alerts tail --limit 30
python -m trace_ml incident list --status open --limit 30
```

Expected:
- Reappearance alerts.
- Existing incident updated instead of many new open incidents for same entity.

---

## 4) Incident + Action Layer Validation (Phase 4)

## A) Operator Severity Control
1. List incidents and copy one full ID:
```powershell
python -m trace_ml incident list --status open --limit 20
```
2. Set severity:
```powershell
python -m trace_ml incident set-severity --id REAL_INCIDENT_ID --severity high
python -m trace_ml incident show --id REAL_INCIDENT_ID
```

Expected:
- Severity changes to `high`.
- Future policy behavior reflects new severity.

## B) Action Audit Visibility
```powershell
python -m trace_ml action list --incident-id REAL_INCIDENT_ID
```

Expected:
- Action records visible with:
  - action type
  - trigger (`on_create` / `on_update`)
  - status
  - timestamp

## C) Incident Lifecycle Control
```powershell
python -m trace_ml incident close --id REAL_INCIDENT_ID
python -m trace_ml incident list --status open --limit 20
python -m trace_ml incident list --status closed --limit 20
```

Expected:
- Incident leaves open list and appears in closed list.

---

## 5) Command-Level Regression Sweep

Run all primary command surfaces once:

```powershell
python -m trace_ml doctor
python -m trace_ml person list
python -m trace_ml person audit
python -m trace_ml train rebuild
python -m trace_ml history query --limit 20
python -m trace_ml report summary
python -m trace_ml report quality
python -m trace_ml export csv
python -m trace_ml events tail --limit 20
python -m trace_ml alerts tail --limit 20
python -m trace_ml incident list --limit 20
```

Pass criteria:
- No command crashes.
- Output tables are readable and meaningful.

---

## 6) Negative / Error-Path Tests

```powershell
python -m trace_ml incident show --id INC-DOES-NOT-EXIST
python -m trace_ml incident close --id INC-DOES-NOT-EXIST
python -m trace_ml action list --incident-id INC-DOES-NOT-EXIST
```

Expected:
- Graceful, explicit error messaging.
- No corruption of existing records.

---

## 7) Load + Soak Test (Practical)

## A) Query Stress (read path)
Open PowerShell and run:
```powershell
1..100 | ForEach-Object { python -m trace_ml history query --limit 50 > $null }
1..100 | ForEach-Object { python -m trace_ml events tail --limit 50 > $null }
1..100 | ForEach-Object { python -m trace_ml alerts tail --limit 50 > $null }
1..50  | ForEach-Object { python -m trace_ml incident list --limit 50 > $null }
```

Expected:
- No crashes, no severe slowdowns.

## B) Live Soak (write path)
- Run `recognize live` continuously for 20-30 minutes.
- During run, vary conditions: lighting, distance, occlusion, enter/exit.
- After run:
```powershell
python -m trace_ml report summary
python -m trace_ml alerts tail --limit 50
python -m trace_ml incident list --limit 50
```

Expected:
- System remains stable.
- Events/alerts/incidents continue to append without corruption.

---

## 8) Final Acceptance Checklist (Must Be All Green)
- [ ] Automated tests pass.
- [ ] Event stream is correct and continuous.
- [ ] Alert rules trigger and are deduplicated (cooldown works).
- [ ] One active incident per entity behavior holds.
- [ ] Incident severity can be changed manually.
- [ ] Actions are policy-driven and logged.
- [ ] Incident close workflow works.
- [ ] Export/report/history commands remain functional.
- [ ] Query stress + live soak complete without crashes.

If all are green, system is fully verified for current architecture scope and ready to move to Service Layer.
