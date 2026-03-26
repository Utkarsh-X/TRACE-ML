# TRACE-ML Improvement Report v2
## Architecture + Modern Stack + CLI→GUI Roadmap
**Updated**: March 23, 2026 | Supersedes v1

---

## What Changed in v2

The original report proposed FAISS for vector search. This version replaces that with **LanceDB + DuckDB** — a more architecturally coherent choice. Additionally, SCRFD replaces RetinaFace as the face detector, and a threaded producer-consumer frame pipeline replaces the synchronous capture loop. A full CLI→GUI progression plan is added in Section 12.

---

## 1. RECOGNITION CORE

### Face Detection — SCRFD (replaces RetinaFace)

Both SCRFD and RetinaFace are from the InsightFace family, but SCRFD (Sample and Computation Redistribution) was specifically designed for constrained CPU deployment.

| | RetinaFace | SCRFD (recommended) |
|---|---|---|
| Detection speed (CPU) | ~15ms/frame | ~5ms/frame |
| Accuracy | Excellent | Excellent (same tier) |
| Model size | ~110MB | 2.5MB (500m variant) |
| Small face handling | Good | Better |

Use `scrfd_500m_bnkps.onnx` for the best speed/accuracy tradeoff. Drop-in via InsightFace's detector API — no extra integration work.

### Face Recognition — ArcFace via InsightFace (unchanged from v1)

`buffalo_sc` model pack: SCRFD detector + MobileNet-based ArcFace recognizer. CPU-optimized, ~85MB download, ships as a single model bundle. Produces 512-D embeddings.

```bash
pip install insightface onnxruntime
```

```python
import insightface
app = insightface.app.FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id=0)  # ctx_id=-1 for GPU if available
faces = app.get(frame)
# faces[i].embedding → 512-D numpy array, already L2-normalised
```

---

## 2. STORAGE ARCHITECTURE — LanceDB + DuckDB

### Why LanceDB Replaces FAISS

FAISS is a pure ANN index — it has no concept of updating or deleting a vector (requires full index rebuild), stores no metadata, and cannot do hybrid queries. LanceDB is an embedded vector database built on the Lance columnar format.

**What you gain with LanceDB:**
- Full CRUD on embeddings — delete a person, their embedding is gone instantly
- Metadata stored alongside vectors (person_id, name, category, confidence)
- Native hybrid search: `similarity > 0.8 AND category = 'criminal'` in one call
- No server, no Docker — just a folder of `.lance` files
- Persists automatically like SQLite

```bash
pip install lancedb
```

```python
import lancedb
import numpy as np

db = lancedb.connect("data/vectors")
table = db.create_table("persons", schema={
    "person_id": str,
    "name": str,
    "category": str,
    "embedding": np.zeros(512, dtype=np.float32)
})

# Register a new person
table.add([{"person_id": "PRC001", "name": "John Doe",
            "category": "criminal", "embedding": embedding_array}])

# Search — returns closest matches with metadata
results = table.search(query_embedding).limit(5).to_pandas()

# Delete — instant, no rebuild
table.delete("person_id = 'PRC001'")
```

### DuckDB as Analytics Layer

DuckDB can query Lance files directly using SQL — no ETL, no sync between two databases. Person metadata and detection history can be queried analytically while LanceDB handles the vector operations.

```bash
pip install duckdb
```

```python
import duckdb

# Query detections directly from Lance files
conn = duckdb.connect()
conn.execute("INSTALL lance; LOAD lance;")

result = conn.execute("""
    SELECT name, category, COUNT(*) as hits, AVG(confidence) as avg_conf
    FROM lance_scan('data/vectors/detections.lance')
    WHERE detected_at > NOW() - INTERVAL '24 hours'
    GROUP BY name, category
    ORDER BY hits DESC
""").fetchdf()
```

**What this enables:**
- Session summary reports with full SQL expressiveness
- Detection history search with date ranges, confidence filters
- Export filtered results to CSV in one query
- Confusion matrix data for model evaluation

### Storage Architecture Summary

```
data/
├── vectors/
│   ├── persons.lance       ← embeddings + person metadata (LanceDB)
│   └── detections.lance    ← detection events with embeddings (LanceDB)
└── detections/
    └── screenshots/        ← JPEG captures on match
```

DuckDB queries `.lance` files directly. No SQLite needed. Single storage format, two query interfaces (vector similarity via LanceDB, SQL analytics via DuckDB).

---

## 3. FRAME PIPELINE — Threaded Producer-Consumer

The current synchronous loop blocks on inference:

```
capture → detect → recognize → display → [wait] → capture ...
```

At dlib speeds (300-600ms/frame), this gives 1-3 FPS. The fix is two threads:

```
Thread A (capture):    frame → queue → frame → queue → frame → queue ...
Thread B (inference):           ↓ detect+recognize         ↓ detect+recognize ...
Main thread (display): render latest result from results_queue
```

```python
import threading
import queue
import cv2

frame_queue = queue.Queue(maxsize=2)   # cap=2 prevents memory buildup
result_queue = queue.Queue(maxsize=2)

def capture_worker(cap):
    while running:
        ret, frame = cap.read()
        if ret:
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass   # drop frame, inference is still busy — intentional

def inference_worker(recognizer):
    while running:
        frame = frame_queue.get()
        detections = recognizer.recognize(frame)
        result_queue.put((frame, detections))

# Start threads
cap = cv2.VideoCapture(0)
threading.Thread(target=capture_worker, args=(cap,), daemon=True).start()
threading.Thread(target=inference_worker, args=(recognizer,), daemon=True).start()

# Main display loop runs at full FPS; inference runs independently
while running:
    if not result_queue.empty():
        frame, detections = result_queue.get_nowait()
        render_hud(frame, detections)
    cv2.waitKey(1)
```

**Result:** Display runs at ~30 FPS. Inference runs as fast as the model allows (5-15 FPS with InsightFace on CPU). No blocking, no dropped display frames.

---

## 4. FULL MODERN STACK (Updated Requirements)

```text
# Recognition
insightface>=0.7.3
onnxruntime>=1.16.0          # CPU; swap onnxruntime-gpu if CUDA

# Storage
lancedb>=0.6.0
duckdb>=0.10.0

# Frame capture
opencv-python>=4.8.0
numpy>=1.24.0

# CLI & TUI
rich>=13.7.0
textual>=0.47.0

# Config & validation
pydantic-settings>=2.1.0
pyyaml>=6.0.1

# Logging
loguru>=0.7.2

# Dev
pytest>=7.4.0
pytest-cov>=4.1.0
ruff>=0.1.0
black>=23.0.0

# GUI (Phase 2 — see Section 12)
# Option A: PyQt6>=6.6.0
# Option B: customtkinter>=5.2.0  (simpler, good for FYP demos)
# Option C: Gradio>=4.0.0         (web-based, easiest to demo remotely)
```

---

## 5. ANTI-SPOOFING — MiniFASNet (Updated from v1)

MiniFASNet is lighter than the Silent-Face library and ships as a clean ONNX model — better for embedding into the pipeline.

- Model: `2.7_80x80_MiniFASNetV2.onnx` (~1.5MB)
- Speed: ~8ms per face crop on CPU
- Output: `real_prob` score 0.0–1.0

```python
class AntispoofChecker:
    def __init__(self, model_path: str, threshold: float = 0.6):
        self.session = ort.InferenceSession(model_path)
        self.threshold = threshold

    def is_real(self, face_crop: np.ndarray) -> tuple[bool, float]:
        inp = self._preprocess(face_crop)
        score = self.session.run(None, {"input": inp})[0][0][1]
        return score > self.threshold, float(score)
```

---

## 6. UPDATED ARCHITECTURE MAP

```
trace_ml/
├── config.yaml
├── requirements.txt
├── README.md
│
├── core/
│   ├── config.py            ← pydantic-settings Config loader
│   ├── logger.py            ← loguru setup
│   └── models.py            ← Pydantic: Person, Detection dataclasses
│
├── detectors/
│   ├── base.py              ← BaseFaceDetector ABC
│   └── scrfd.py             ← SCRFDDetector (InsightFace)
│
├── recognizers/
│   ├── base.py              ← BaseRecognizer ABC
│   └── arcface.py           ← ArcFaceRecognizer (InsightFace buffalo_sc)
│
├── store/
│   ├── vector_store.py      ← LanceDB wrapper (CRUD + hybrid search)
│   └── analytics.py         ← DuckDB query helpers
│
├── pipeline/
│   ├── capture.py           ← Threaded capture worker
│   ├── inference.py         ← Threaded inference worker
│   ├── collect.py           ← Face collection with Rich progress
│   ├── train.py             ← Embedding registration pipeline
│   ├── session.py           ← Recognition session orchestrator
│   └── antispoof.py         ← MiniFASNet wrapper
│
├── cli/                     ← Phase 1: Terminal UI
│   ├── main.py              ← Textual app entry point
│   ├── dashboard.py         ← Live session TUI screen
│   ├── registry.py          ← Person management TUI screen
│   └── reports.py           ← Rich-formatted session reports
│
├── gui/                     ← Phase 2: Desktop GUI (see Section 12)
│   ├── app.py               ← Main GUI entry point
│   ├── views/
│   │   ├── dashboard.py     ← Live feed + detection panel
│   │   ├── registry.py      ← Person management
│   │   ├── history.py       ← Detection history + search
│   │   └── settings.py      ← Config editor
│   └── widgets/
│       ├── camera_feed.py   ← OpenCV → GUI frame widget
│       └── alert_card.py    ← Detection alert component
│
└── tests/
    ├── test_recognizer.py
    ├── test_vector_store.py
    └── test_pipeline.py
```

---

## 7. IMPLEMENTATION ORDER (Updated)

**Phase 1 — Stabilize (2–3 days)**
1. Fix `train_model.py` merge conflict
2. Add `requirements.txt` and `config.yaml`
3. Add loguru — wrap all I/O in `@logger.catch`
4. Add pydantic Config loader

**Phase 2 — New Core (4–5 days)**
5. InsightFace recognizer + SCRFD detector
6. LanceDB vector store (CRUD + search)
7. DuckDB analytics helpers
8. Threaded capture + inference pipeline

**Phase 3 — CLI Polish (2–3 days)**
9. Rich tables, progress bars, panels
10. Session summary report
11. Detection history search + CSV export

**Phase 4 — Differentiators (3–4 days)**
12. MiniFASNet anti-spoofing
13. Model evaluation report (metrics, confusion matrix)
14. Batch image import
15. ABC architecture + plugin registry

**Phase 5 — GUI (4–6 days, see Section 12)**
16. Choose framework (recommendation below)
17. Live feed view with HUD overlay
18. Person registry CRUD view
19. Detection history / analytics view
20. Settings panel wired to config.yaml

**Phase 6 — Quality (2 days)**
21. Unit tests (repositories, recognizers)
22. README + architecture diagram
23. Demo script for presentation

**Total: ~3–4 weeks working consistently**

---

## 8. WHAT TO CLAIM IN YOUR PROJECT REPORT

| Claim | Implementation backing |
|---|---|
| State-of-the-art recognition (ArcFace) | InsightFace buffalo_sc |
| SCRFD multi-scale face detection | InsightFace SCRFD |
| Embedded vector database with CRUD | LanceDB |
| Hybrid vector+metadata search | LanceDB filter queries |
| SQL analytics over vector data | DuckDB + Lance scan |
| Anti-spoofing liveness detection | MiniFASNet ONNX |
| Concurrent capture/inference pipeline | Python threading + Queue |
| Plugin-based recognizer architecture | ABC + registry pattern |
| Typed configuration system | pydantic-settings |
| Structured audit-grade logging | loguru |
| Model evaluation with metrics | Hold-out set + seaborn confusion matrix |

---

## 12. CLI → GUI PROGRESSION PLAN

### 12.1 Why Bother with a GUI for a Final Year Project?

CLI is perfectly defensible, but a GUI changes how an examiner *experiences* the project. A live demo where detections appear as cards in a sidebar while the camera feed shows bounding boxes and confidence scores is a different category of impression than watching a terminal scroll. It also directly demonstrates your software engineering breadth.

The strategy: build CLI first (it's the backbone), then wrap it with a GUI that calls the same underlying pipeline modules. Zero duplicate logic.

### 12.2 Framework Recommendation

**CustomTkinter** — best fit for this project.

| Framework | Verdict | Why |
|---|---|---|
| PyQt6 | ✅ Powerful, ❌ complex | Overkill for FYP; signals/slots model has steep curve |
| Tkinter (raw) | ❌ Ugly | Default widgets look like 1995 |
| CustomTkinter | ✅ Recommended | Modern dark UI out of the box, embeds OpenCV easily, pure Python, ships in one pip install |
| Gradio | ✅ for remote demos | Web-based, great for sharing; not ideal for real-time video |
| Dear PyGui | ✅ Fast, GPU-rendered | Excellent for real-time, slightly unusual API |

CustomTkinter gives you: dark/light theme, modern rounded widgets, works everywhere Python works, and an OpenCV camera frame can be rendered directly into a `CTkLabel` by converting via PIL. That's the core integration trick.

### 12.3 Five Core GUI Views

**View 1 — Live Recognition Dashboard**
The centrepiece. Split layout: left 65% is the live camera feed with OpenCV HUD drawn over it (bounding boxes, name labels, confidence bars). Right 35% is a scrolling alert feed — each detection is a card showing name, category badge, timestamp, confidence percentage, and a thumbnail. Session stats bar at top: elapsed time, frame count, detection count, unique persons seen.

**View 2 — Person Registry**
A searchable table of all registered persons with columns: ID, name, category (Criminal/Missing), DOB, location, # training images, # past detections. Row click opens a detail panel: full metadata form (editable), training image thumbnails, detection history timeline. Add new person button opens a registration wizard that invokes the collection pipeline directly — webcam capture inside the GUI window.

**View 3 — Detection History**
Full detection log with filters: date range picker, person filter dropdown, confidence threshold slider, category toggle. Results shown as a table with sortable columns. Clicking a row shows the screenshot taken at detection time. Export button runs DuckDB query and saves filtered results to CSV. Summary statistics panel: total detections, unique persons, avg confidence, top 5 most detected.

**View 4 — Model & Analytics**
Training controls: train/retrain button with progress bar, current model stats (persons registered, embeddings count, last trained timestamp). Evaluation panel: run against test set and display precision/recall per identity, overall accuracy, false positive rate. Confusion matrix heatmap rendered via matplotlib embedded in the GUI. Export evaluation report to PDF.

**View 5 — Settings**
Live-editable config panel wired to config.yaml. Sliders for confidence threshold and similarity threshold with live preview of what current setting means ("at 0.65, expect ~2% false positives"). Camera selector dropdown. Algorithm selector (if multiple recognizers registered). Anti-spoofing toggle. Log level selector.

### 12.4 OpenCV → CustomTkinter Integration Pattern

```python
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import threading

class CameraWidget(ctk.CTkLabel):
    def __init__(self, master, pipeline, **kwargs):
        super().__init__(master, text="", **kwargs)
        self.pipeline = pipeline
        self._running = True
        threading.Thread(target=self._update_loop, daemon=True).start()

    def _update_loop(self):
        while self._running:
            frame, detections = self.pipeline.get_latest()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                self.configure(image=img)
                self._image = img  # keep reference
            self.after(33)  # ~30 FPS target

    def stop(self):
        self._running = False
```

The pipeline (threaded producer-consumer) runs independently. The widget polls it at 30 FPS and renders whatever frame is latest. Detections from the same poll call drive the alert feed in the sidebar — same data, two visual representations.

### 12.5 GUI Implementation Order

After the CLI phases are complete:

1. Set up CustomTkinter app shell with nav sidebar and view routing (1 day)
2. Camera widget — embed OpenCV frame with HUD overlay (1 day)
3. Alert feed sidebar — detection cards from recognition session (1 day)
4. Person registry table + detail panel + registration wizard (2 days)
5. Detection history with filters + DuckDB-backed export (1 day)
6. Settings panel wired to config.yaml with live reload (1 day)
7. Model analytics view + matplotlib confusion matrix (1 day)

**Total GUI: ~8 days**

---

*End of Report — TRACE-ML v2 | CLI→GUI Architecture + Modern Stack*