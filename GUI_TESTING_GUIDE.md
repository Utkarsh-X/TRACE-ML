# TRACE-AML GUI Testing Guide

## System Status: ✅ FRESH DATABASE RESET

All data cleaned. Ready for fresh user workflow testing through the web interface.

---

## 🚀 QUICK START

### Step 1: Start the Service

```powershell
# Activate venv if not already active
cd "d:\github FORK\TRACE-ML"
& ".venv311\Scripts\Activate.ps1"

# Start the service (FastAPI + Uvicorn on port 8080)
python start_service.py
```

**Expected Output:**
```
╭────────────────────────────────────────────────╮
│ TRACE-AML Service Startup                      │
│ FastAPI + Uvicorn                              │
╰────────────────────────────────────────────────╯

🚀 Starting Uvicorn on http://0.0.0.0:8080

📘 Access the UI at: http://localhost:8080/ui/live_ops/index.html
📘 API docs: http://localhost:8080/docs

INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### Step 2: Access the GUI

Open browser and navigate to:
```
http://localhost:8080/ui/live_ops/index.html
```

You should see a blank **Live Ops** dashboard with sections for:
- Active Entities (empty initially)
- Active Incidents (empty)
- Recent Alerts (empty)
- System Health metrics

---

## 📋 FULL ENROLLMENT CYCLE TEST

### PHASE 1: Person Registration

**Objective:** Create a new person record

#### Navigation
From the Live Ops page, click **"Enrollment"** in the left sidebar to go to:
```
http://localhost:8080/ui/enrollment/index.html
```

#### What You'll See
```
┌─ LEFT SIDEBAR ──────┬─ CENTER (Image Upload) ─┬─ RIGHT (Training) ──┐
│ (Empty initially)   │ Drop zone for images    │ Training status     │
│                     │ Form fields:            │ (not trained yet)   │
└─────────────────────┴────────────────────────┴─────────────────────┘
```

#### Create Person #1: john

1. **Fill In the Form:**
   - **Name:** John Doe
   - **Category:** criminal (dropdown)
   - **Severity:** high (optional)
   - **DOB:** 1985-03-15 (optional)
   - **Gender:** Male (optional)
   - **City:** New York (optional)
   - **Country:** USA (optional)
   - **Notes:** Test person for enrollment (optional)

2. **Click "Create Person"**

**Expected Result:**
- Person card appears in left sidebar showing:
  ```
  PRC001
  John Doe
  criminal | imgs: 0 | emb: 0
  ```
- State badge shows: **DRAFT** (yellow)
- Center panel resets for image upload

#### Upload Reference Images

3. **Upload 8-10 images:**
   - Click "Click to browse" or drag-drop images to drop zone
   - Images should be:
     - Good lighting
     - Clear frontal face
     - Different angles (some profile, some straight-on)
     - Different expressions (neutral, smiling)
   - **Minimum 6 images required for quality gate**

**Expected Result:**
- Thumbnail grid shows uploaded images
- File count displays: "8 images selected"
- Images saved to: `data/person_images/PRC001/`

4. **Right Panel Monitor:**
   - Image count updates as upload progresses
   - Quality assessment runs on backend

---

### PHASE 2: Embedding Training & Lifecycle Transition

**Objective:** Train embeddings and transition draft → ready → active

#### Trigger Training

5. **On the Enrollment page, right sidebar:**
   - Click **"Trigger Training"** button

**Expected Behavior:**
- Button becomes disabled (training in progress)
- Status message: "Training in progress..."
- Training polls every 2 seconds to check status

#### Monitor Training Progress

**What happens behind the scenes:**

1. **Quality Assessment** (per-image):
   - Sharpness: checks if image is sharp
   - Face Ratio: ensures face fills 3%+ of image
   - Brightness: validates lighting (45-220 range)
   - Pose Score: checks face is frontal-ish (>0.28)
   - Images with score ≥ 0.38 marked as VALID

2. **Embedding Generation** (per-valid-image):
   - ArcFace model generates 512-dim vectors
   - Embeddings stored in `data/vectors/person_embeddings/`

3. **Lifecycle Update:**
   - **READY**: When ≥2 embeddings AND ≥6 valid images
   - **ACTIVE**: When ≥6 embeddings AND ≥6 valid images

#### Check Results

6. **Training Complete (30-60s later):**

**Expected Status Display:**
```
Enrollment Status:
  Score: 0.85 (enrollment_score)
  State: ACTIVE (green badge)
  Reason: "ready_for_live_deployment"
  Images: 8 valid / 8 total
  Embeddings: 8
```

7. **Verify in Person Card (left sidebar):**
```
PRC001
John Doe
criminal | imgs: 8 | emb: 8  ← counts updated
```
State badge: **ACTIVE** (green)

---

### CREATE 2-3 MORE PERSONS

Repeat **PHASE 1-2** for additional test persons:

**Person #2: alice**
- Category: vip
- Upload 8-10 different images (different face)

**Person #3: bob**
- Category: missing
- Upload 8-10 different images (different face)

#### Expected Enrollment Panel After All 3:
```
┌ LEFT SIDEBAR ──────────┐
│ PRC001        [ACTIVE] │
│ John Doe               │
│ criminal | imgs: 8     │
│                        │
│ PRC002        [ACTIVE] │
│ Alice Smith            │
│ vip | imgs: 8          │
│                        │
│ PRC003        [ACTIVE] │
│ Bob Johnson            │
│ missing | imgs: 8      │
└────────────────────────┘
```

---

## 🎥 LIVE RECOGNITION TESTING

### PHASE 3: Enable Camera & Start Recognition

**Objective:** Test real-time face detection and recognition

#### Navigation

1. Go back to **Live Ops**:
```
http://localhost:8080/ui/live_ops/index.html
```

#### Camera Control Panel

Look for the **"Camera Control"** section (top-left of the page):
```
┌─ Camera Control ──────────┐
│                           │
│ Status: [Disabled]        │
│                           │
│ [Enable Camera]           │
│ [Enable Recognition]      │
│                           │
└───────────────────────────┘
```

#### Enable Camera

2. **Click "Enable Camera"**

**Expected Result:**
- Button changes to "Disable Camera" (red)
- Status message: "Camera enabled"
- Webcam light turns on (if visible)
- Console logs: "Camera started successfully"

#### Enable Face Recognition

3. **Click "Enable Recognition"**

**Expected Result:**
- Button changes to "Disable Recognition" (red)
- Status message: "Face recognition started"
- System begins processing frames

### PHASE 4: Test Known Person Detection

**Objective:** Verify enrolled person is recognized

#### Show Known Person to Camera

4. **Position camera to see John Doe (or whichever person you enrolled)**
   - Face should be clearly visible
   - Good lighting
   - Frontal orientation (not extreme profile)

#### Observe Live Dashboard

**You should see updates in real-time:**

**Active Entities section:**
```
PRC001 (John Doe)
Confidence: 85.3%
Category: criminal
Last Seen: just now
```

**System Health (footer):**
```
Entities: 1
Incidents: 0
Alerts: 0
Detections: 15
FPS: 28.5
```

**Expected Metrics:**
- Confidence rises as face stays in view (temporal smoothing)
- Decision state: **ACCEPT** (green confidence >= 0.72)
- Multiple detections logged (one per frame)

#### Temporal Smoothing Demo

5. **Watch confidence evolve:**
   - Frame 1-2: High variance (raw confidence)
   - Frame 3-6: Smooth trend upward (EMA smoothing alpha=0.6)
   - After 6 frames: Stable high confidence (voting window full)

---

### PHASE 5: Test Unknown Person Detection

**Objective:** Verify unknown face handling and clustering

#### Show Unknown Face to Camera

6. **Position camera to show a different face (not in database)**
   - Can be you, a colleague, anyone not enrolled
   - Let face stay visible for 5+ seconds

#### Observe Unknown Entity Creation

**Active Entities section updates:**
```
UNK001
Confidence: 42.1%
Category: unknown
Last Seen: just now
```

#### Key Observations:

- **Min Surface Threshold:** Unknown must reach 35.0% confidence to be created
- **Unknown Aggregation:** If same face appears again, may reuse UNK001 (similarity > 0.55)
- **No False Matches:** Should NOT match to enrolled persons

---

### PHASE 6: Test Quality-Dependent Thresholds

**Objective:** Verify robust matching adapts to image quality

#### Low-Quality Test

7. **Position known person in challenging conditions:**
   - Side profile (not frontal)
   - Dim lighting
   - Far from camera
   - Partially occluded

#### Expected Behavior:

- **Threshold Relaxation:** Accept threshold may drop (0.72 → 0.60)
- **Confidence Drops:** Raw confidence lower than good-quality frames
- **Decision:** May show **REVIEW** instead of ACCEPT
- **Robust Matching:** Enabled by default (robust_matching=true)

#### High-Quality Test

8. **Optimal conditions:**
   - Frontal face
   - Bright even lighting
   - Close to camera
   - No occlusion

**Expected Result:**
- Confidence: 85-95%
- Decision: **ACCEPT** immediately
- Very stable smoothed confidence

---

## 📊 INTELLIGENCE PIPELINE TESTS

### PHASE 7: Trigger Alert Rules

**Objective:** Verify intelligent alert generation

#### Test Reappearance Rule

9. **Known person workflow:**
   - Show John Doe to camera (5+ seconds)
   - **Leave frame completely** (30+ seconds)
   - **Return to frame**

**Expected Alert:**
```
Alerts section updates with:
┌─────────────────────────────────────────┐
│ REAPPEARANCE                    [HIGH]  │
│ Entity: PRC001 (John Doe)               │
│ Time: 2026-04-13 21:47:32 UTC           │
│ Rule: Reappearance after 30+ seconds    │
└─────────────────────────────────────────┘
```

**Incident Created:**
```
Incidents section:
┌─────────────────────────────────────────┐
│ #INC001                      [OPEN]     │
│ Entity: PRC001 (John Doe)               │
│ Severity: HIGH                          │
│ Alerts: 1                               │
│ Created: 2026-04-13 21:47:32 UTC        │
└─────────────────────────────────────────┘
```

#### Test Unknown Recurrence Rule

10. **Unknown person workflow:**
    - Unknown face visible (5+ seconds)
    - **Leave frame**
    - **Return within 10 seconds**

**Expected Alert:**
```
UNKNOWN_RECURRENCE                 [MEDIUM]
Entity: UNK001
Reason: Same unknown face reappeared
```

#### Test Instability Rule

11. **High-jitter detection:**
    - Position person at camera edge
    - Move back and forth slightly
    - Track will show high confidence variance

**Expected Alert:**
```
INSTABILITY                         [MEDIUM]
Entity: PRC001
Reason: Confidence variance > 0.15 in 10s window
```

---

## 🔍 TIMELINE & HISTORY TESTS

### PHASE 8: Review Forensic Timeline

**Objective:** Verify event and detection logging

#### Navigate to History

12. **Click "History" in sidebar:**
```
http://localhost:8080/ui/history/index.html
```

#### Expected Dashboard:

**Timeline View:**
```
21:47:32 ALERT      → REAPPEARANCE HIGH
21:47:30 EVENT      → Detection: PRC001 confidence 85.2%
21:47:28 EVENT      → Detection: PRC001 confidence 84.8%
21:47:25 ALERT      → UNKNOWN_RECURRENCE MEDIUM
...
```

**Query Options:**
- Filter by entity type (known vs unknown)
- Filter by time range (last hour, last 24h, etc.)
- Filter by severity

#### Verify Detections CSV Export

13. **Click "Export Detections"**

**Expected Result:**
- File generated: `data/exports/detections_YYYYMMDDTHHMMSSZ.csv`
- Columns: timestamp, person_id, name, category, confidence, similarity, decision
- Rows: all detections from session

---

## ✅ SUCCESS CRITERIA

### Enrollment Phase:
- [ ] 3+ persons created
- [ ] All in ACTIVE state after training
- [ ] Image counts show 6-10 per person
- [ ] Embedding counts >= 6

### Recognition Phase:
- [ ] Camera enables without errors
- [ ] Recognition processes frames (FPS > 20)
- [ ] Known person detected with high confidence (>0.72)
- [ ] Unknown person creates entity (if confidence > 35%)
- [ ] Confidence smoothing visible (reduces jitter)

### Intelligence Phase:
- [ ] Alerts generated for rule violations
- [ ] Incidents created and tracking alerts
- [ ] Timeline logs events chronologically
- [ ] Detection export works

### Quality Phase:
- [ ] High-quality faces: ACCEPT (confidence > 0.72)
- [ ] Low-quality faces: REVIEW or REJECT
- [ ] Threshold relaxation visible in logs

---

## 🐛 DEBUGGING TIPS

### View Service Logs

Terminal where service is running shows:
```
INFO:     127.0.0.1:54321 "GET /api/v1/live/snapshot HTTP/1.1" 200
INFO:     127.0.0.1:54322 "POST /api/v1/camera/enable HTTP/1.1" 201
```

### Database Inspection

```powershell
# Check persons in database
python -c "
from trace_aml.store.vector_store import VectorStore
from trace_aml.core.config import load_settings
settings = load_settings()
store = VectorStore(settings)
persons = store.list_persons()
for p in persons:
    print(f'{p[\"person_id\"]}: {p[\"name\"]} ({p[\"lifecycle_state\"]})')
"
```

### Check Browser Console

Press F12 in browser for developer tools:
- **Console tab:** JavaScript errors
- **Network tab:** API response times
- **Application tab:** Local state

### Common Issues

| Issue | Solution |
|-------|----------|
| "Cannot find device 0" | Check camera is detected: `python -m trace_aml doctor` |
| Health checks failing | Run health check: `python -m trace_aml doctor` |
| "camera not enabled" error | Enable camera first, then recognition |
| API returns 500 | Check service logs in terminal |
| No entities appearing | Faces too low quality or below min_unknown_surface_threshold (35%) |

---

## 📌 KEY THRESHOLDS (from config.yaml)

These control behavior you'll observe:

**Recognition:**
- `accept_threshold: 0.72` - Confidence must exceed this for ACCEPT decision
- `review_threshold: 0.58` - Between review and reject
- `similarity_threshold: 0.45` - Minimum match score to database

**Quality:**
- `min_quality_score: 0.38` - Image must pass this to be counted as "valid"
- `min_face_ratio: 0.03` - Face must fill 3%+ of image
- `min_sharpness: 55.0` - Image sharpness (Laplacian)
- `min_brightness: 45; max: 220` - Lighting range

**Temporal:**
- `decision_window: 6` - Voting window size (6 frames)
- `smoothing_alpha: 0.6` - EMA smoothing factor
- `min_commit_confidence: 45.0` - Must reach 45% before DB write (prevents ghosts)

**Rules:**
- `cooldown_sec: 15` - Wait 15s before firing same rule again
- `reappearance`: 10s window, 3 events min
- `unknown`: 10s window, 3 events min
- `instability`: 10s window, std_dev > 0.15

**Unknown:**
- `unknown_reuse_threshold: 0.55` - Unknown cluster similarity threshold
- `min_unknown_surface_threshold: 35.0` - Min confidence to create UNK entity

---

## ⏱️ EXPECTED TIMINGS

| Operation | Time | Notes |
|-----------|------|-------|
| Person creation | <1s | Just DB insert |
| Image upload (10 images) | 5-10s | Network + file save |
| Training (10 images) | 30-60s | Quality assessment + embedding |
| Camera startup | 2-3s | OpenCV initialization |
| First detection | 3-5s | Model warmup |
| Confidence stabilization | 2-3s (6 frames) | 6-frame voting window |
| Alert generation | <1s | After rule fires |

---

## 🎯 NEXT STEPS AFTER SUCCESS

Once all phases pass:
1. **Document any bugs found** with:
   - Expected behavior
   - Actual behavior
   - Reproduction steps
   - Relevant logs/screenshots

2. **Performance baseline:**
   - Average FPS
   - Average detection latency
   - Database query times

3. **Integration readiness:**
   - Electron wrapper testing
   - Multi-user concurrent access
   - Long-running stability (8+ hours)

---

## 📄 Document History

- **Created:** 2026-04-13
- **Status:** Fresh system reset, ready for GUI testing
- **Database:** Completely clean, zero records
- **Next Action:** Start service and begin enrollment workflow
