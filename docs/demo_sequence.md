# TRACE-ML v3 Demo Sequence

## Objective
Show a complete end-to-end flow on a single laptop webcam in under 5 minutes.

## Pre-demo checklist
- Confirm dependencies are installed.
- Confirm webcam is free (not occupied by another app).
- Confirm at least one registered person has images.

## Command runbook
```powershell
trace-ml doctor
trace-ml person list
trace-ml train rebuild
trace-ml recognize live
trace-ml history query --limit 10
trace-ml report summary
trace-ml export csv
```

## Presenter notes
- During `recognize live`, press `q` to end the session.
- Highlight tactical HUD panel, confidence scores, and automatic detection persistence.
- In `history query`, explain confidence filtering and retrieval.
- In `report summary`, explain top detections and average confidence.
