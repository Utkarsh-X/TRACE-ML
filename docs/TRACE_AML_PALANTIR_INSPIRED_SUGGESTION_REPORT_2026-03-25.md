# TRACE-AML Palantir-Inspired Suggestion Report
Date: 2026-03-25  
Project Intent: Build a best-in-class criminal-entity tracking and recognition platform with strong operational intelligence workflows.

## 1. Positioning Recommendation
TRACE-AML should be positioned as an intelligence-support system for public safety operations, not as a fully autonomous decision engine.

Core message:
1. Human-in-the-loop decisions.
2. High explainability and auditability.
3. Fast triage and operational clarity under uncertainty.

Naming note:
1. `TRACE-AML` is strong and future-facing.
2. Be aware `AML` is widely associated with Anti-Money Laundering in industry contexts.
3. Suggested project subtitle for clarity: "Criminal Entity Recognition and Tracking Intelligence Platform."

## 2. Palantir-Like Design Principles to Adopt
1. Ontology-first thinking: persons, detections, incidents, cases, operators, alerts.
2. Workflow over widgets: operators should complete tasks, not just view outputs.
3. Every alert must explain "why now" with traceable decision metadata.
4. Multi-layer confidence: model confidence, temporal confidence, and policy confidence.
5. Decision accountability: immutable logs and operator acknowledgments.

## 3. Strategic Capability Blueprint
### Pillar A: Command and Control CLI
1. `ops live` tactical command center.
2. Threat score strip and incident counters.
3. Active case panel and alert feed.

### Pillar B: Case and Incident Intelligence
1. `case create/list/show/close`.
2. `incident open/list/assign/close`.
3. Detection-to-case linking with notes and annotations.

### Pillar C: Policy and Alerting
1. `rules.yaml` for configurable escalation logic.
2. Conditions based on accept streak, review spikes, unknown recurrence, and confidence trends.
3. Actions such as alert raise, screenshot bundle, incident creation.

### Pillar D: Unknown Entity Intelligence
1. Unknown clustering into stable IDs (`UNK-001`).
2. Reappearance analytics by timestamp and track pattern.
3. Unknown-to-known promotion workflow when identified later.

### Pillar E: Evidence and Forensics
1. Incident-level export packages.
2. Include timeline, confidence series, screenshots, and metadata manifest.
3. Optional hashing/signatures for tamper-evident evidence chains.

### Pillar F: Governance and Responsible AI Controls
1. Role-based access boundaries for sensitive operations.
2. Explicit retention and redaction policy.
3. Operator acknowledgment and reason codes for critical actions.

## 4. Recommended Roadmap (CLI-First to GUI)
## Phase 1 (Immediate, 1-2 weeks)
1. Build case/incident domain in storage and CLI.
2. Add rules engine and alert command set.
3. Add unknown clustering and unknown report.
4. Upgrade reports with operational KPIs.

## Phase 2 (2-4 weeks)
1. Add replay mode from recorded sessions for deterministic evaluations.
2. Add benchmark command for stress-test scenarios.
3. Add audit log with append-only event records.

## Phase 3 (GUI Transition)
1. Expose core as service endpoints (REST + WebSocket stream).
2. Keep CLI as first-class operator client.
3. Build GUI as thin layer over identical services.

## 5. Technical Enhancements Recommended Next
1. Add `alerts` table and `incidents` table in LanceDB.
2. Add `case_id` linkage in detection decisions.
3. Add unknown embedding clustering service in analytics layer.
4. Add `trace-ml ops live` command for a focused command-center UX.
5. Add `trace-ml incident export --incident-id ...` command.

## 6. KPI Model for "Best-in-Class" Progress
Use these as formal engineering KPIs:
1. Detection stability: accept/review jitter per minute.
2. False escalation rate: alerts with no incident significance.
3. Mean time to triage: first alert to incident assignment.
4. Unknown reidentification quality: cluster purity and recurrence recall.
5. Operator trust score proxy: percentage of accepted alerts after review.

## 7. Risk Register and Mitigations
1. Overfitting to a single subject or lighting condition.
Mitigation: scenario replay suite + profile-based testing matrix.

2. Alert fatigue from noisy review events.
Mitigation: policy thresholds, cooldowns, and severity rankers.

3. Governance gaps in surveillance context.
Mitigation: immutable logs, explicit permissions, retention limits.

4. Premature GUI complexity.
Mitigation: complete domain workflows in CLI before UI expansion.

## 8. Recommended Defaults for Next Sprint
Policy defaults:
1. `balanced` escalation policy.
2. Both criminal tracking and unknown-entity monitoring enabled.
3. CLI-first delivery with API-ready boundaries.

Priority implementation order:
1. Case + incident data model and commands.
2. Rules engine with alert generation.
3. Unknown clustering and recurrence report.
4. Ops dashboard command and evidence export.

## 9. Proposed Sprint-1 Deliverables
1. New CLI groups: `case`, `incident`, `alert`, `unknown`, `ops`.
2. New persistence tables: `cases`, `incidents`, `alerts`, `unknown_clusters`.
3. Report extensions: incident summary and alert severity distribution.
4. Test suite additions for case lifecycle, alert trigger logic, and unknown clustering stability.
5. Demo script showing full operator workflow from detection to case closure.

## 10. Final Recommendation
The current architecture is ready to evolve into a true intelligence operations stack.  
Do not jump directly to GUI. Build the operational ontology and workflows in CLI first, then expose them through services for a polished interface layer.  
This sequence will produce a system that feels genuinely "best-in-class" for project scope while staying technically defensible.

## 11. Prioritized Execution Matrix
Priority plan:
1. `P0` (must-do): case + incident model, alert rules engine, unknown clustering baseline, incident export package.
2. `P1` (high-value): ops command center mode, severity scoring, operator acknowledgments.
3. `P2` (next stage): API/WebSocket boundary, GUI adapter layer, governance hardening.

Indicative effort:
1. P0: 1-2 focused sprints.
2. P1: 1 sprint.
3. P2: 1-2 sprints.

## 12. Definition of Done for Next Phase
Minimum completion criteria before GUI:
1. Cases and incidents are first-class entities in storage and CLI.
2. Rule-based alerts can be configured without code edits.
3. Unknown entity recurrence is trackable via stable unknown IDs.
4. Incident export includes timeline, evidence assets, and decision trace metadata.
5. End-to-end tests cover rule firing, incident lifecycle, and unknown clustering.

## 13. Scope Guardrails (Do Not Overbuild Yet)
1. Do not implement multi-camera orchestration before case/incident workflows.
2. Do not build heavy frontend before API contracts are stabilized.
3. Do not overfit on cosmetic UX while alert logic is still immature.
4. Do not claim autonomous decision authority; keep review-centric posture.

## 14. Final Strategy Verdict
Planning verdict:
1. Suggestion report is now complete enough to guide development and viva discussion.
2. Roadmap balances ambition and feasibility for a final-year project timeline.
3. Sequence is technically safe: intelligence workflows first, GUI second.

Go/No-Go:
1. `Green` to execute P0 roadmap immediately.
