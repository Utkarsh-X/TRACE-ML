"""
TRACE-AML UI/UX Enhancement Plan

Addresses user feedback from first testing session:
1. Profile photo system (best recognition auto-capture)
2. In-app camera capture for enrollment
3. Location validation (countries + city autocomplete)
4. Date of birth validation + picker
"""

# ============================================================================
# ISSUE 1: PROFILE PHOTO SYSTEM
# ============================================================================
"""
PROBLEM: Users have no visual profile stored. Enrollment successful but 
         person lookup shows no face.

SOLUTION: 
  1. Add profile_photo_path field to PersonRecord
  2. Auto-capture image at highest recognition confidence (>60%)
  3. Store best recognition moment as profile picture
  4. Display in person cards and listings

IMPLEMENTATION:
  - Backend: Endpoint to update profile photo
  - Pipeline: When detection.smoothed_confidence > PROFILE_THRESHOLD,
              compare with existing profile and swap if better
  - Frontend: Display profile thumbnails in person lists
"""

PROFILE_PHOTO_CONFIG = {
    "min_confidence_threshold": 0.60,  # Only save if confidence >= 60%
    "auto_update": True,               # Auto-update to better match
    "max_profile_filesize": 500_000,   # 500KB max
}

PROFILE_PHOTO_SCHEMA_EXTENSION = """
ALTER TABLE persons ADD COLUMN profile_photo_path STRING;
ALTER TABLE persons ADD COLUMN profile_photo_confidence FLOAT;
ALTER TABLE persons ADD COLUMN profile_photo_updated_at STRING;
"""


# ============================================================================
# ISSUE 2: IN-APP CAMERA CAPTURE FOR ENROLLMENT  
# ============================================================================
"""
PROBLEM: Users must have pre-existing images. No way to capture directly
         from camera during enrollment.

SOLUTION:
  1. Add camera capture UI component to enrollment form
  2. Live preview of camera feed
  3. Real-time quality feedback (sharpness, brightness, pose)
  4. "Snap" button to capture high-quality frames
  5. Captured images automatically added to upload list

FRONTEND IMPLEMENTATION (enrollment.js enhancement):
  - Add camera capture section (below or alongside drop-zone)
  - Show live video stream with pose/quality overlay
  - Quality feedback: GREEN (good), YELLOW (acceptable), RED (poor)
  - Capture button with auto-validation
  - Auto-add captured image to selected files array

BACKEND SUPPORT:
  - Existing /api/v1/persons/{person_id}/images endpoint already handles uploads
  - Add /api/v1/quality/assess endpoint for real-time feedback
    Input: raw frame (JPEG)
    Output: {sharpness, brightness, face_ratio, pose_score, overall_pass}
  - Add /api/v1/detection/quick endpoint for instant face detection
    Input: raw frame (JPEG)
    Output: {bbox, detector_score} for visualization
"""

CAMERA_CAPTURE_UI = """
┌─ ENROLLMENT FORM ────────────────────────────────────────┐
│                                                           │
│ [Person Form] [Image Upload] [✨ Camera Capture]        │
│                                                           │
│ ┌─ CAMERA CAPTURE SECTION ─────────────────────────┐    │
│ │ ✓ Request camera permission                      │    │
│ │ ┌─────────────────────────────────────────────┐  │    │
│ │ │ [Live Video Feed]                           │  │    │
│ │ │ Face detected: CENTER                        │  │    │
│ │ │ Quality: ████████░░ 85% (GOOD)              │  │    │
│ │ │ - Sharpness: ✓ (78)                         │  │    │
│ │ │ - Brightness: ✓ (145)                       │  │    │
│ │ │ - Face Ratio: ✓ (18%)                       │  │    │
│ │ │ - Pose: ✓ (0.82)                            │  │    │
│ │ └─────────────────────────────────────────────┘  │    │
│ │                                                    │    │
│ │ [📸 Capture Image] [❌ Close Camera]             │    │
│ │                                                    │    │
│ │ Captured: 3 images (ready to upload)             │    │
│ └────────────────────────────────────────────────────┘   │
│                                                           │
│ [Create Person & Upload]                                │
└─────────────────────────────────────────────────────────┘
"""

QUALITY_FEEDBACK_THRESHOLDS = {
    "sharpness": {"good": 70, "acceptable": 55},      # Laplacian variance
    "brightness": {"good": (80, 180), "acceptable": (45, 220)},
    "face_ratio": {"good": 0.15, "acceptable": 0.03},  # 15% vs 3% of image
    "pose_score": {"good": 0.6, "acceptable": 0.28},
}


# ============================================================================
# ISSUE 3: LOCATION VALIDATION & AUTOCOMPLETE
# ============================================================================
"""
PROBLEM: Free-text city/country fields allow invalid entries ("asdfgh").
         No validation or suggestions.

SOLUTION:
  1. Country field: Dropdown with ISO 3166 countries
  2. City field: Autocomplete from GeoNames database
  3. Auto-suggestions as user types
  4. Server-side validation
  5. Optional: Store latitude/longitude for future mapping

FRONTEND IMPLEMENTATION:
  - Country: <select> with all ISO countries (hard-coded list or from API)
  - City: <input type="text"> with datalist or autocomplete library
  - On country change: Update city suggestions
  - On city select: Store {city, country, lat, lon}

BACKEND IMPLEMENTATION:
  - GET /api/v1/geo/countries → List of {code, name, flag}
  - GET /api/v1/geo/cities?country=US → List of {name, lat, lon}
  - GET /api/v1/geo/cities?query=new&country=US → Autocomplete
  - POST /api/v1/geo/validate → Validate city exists in country
  
  Data source options:
    a) GeoNames.org API (free with registration)
    b) Offline database (countries_cities.json)
    c) Simple hardcoded list (USA, UK, Canada only for MVP)
"""

COUNTRIES_DATASET = [
    {"code": "US", "name": "United States", "flag": "🇺🇸"},
    {"code": "GB", "name": "United Kingdom", "flag": "🇬🇧"},
    {"code": "CA", "name": "Canada", "flag": "🇨🇦"},
    {"code": "AU", "name": "Australia", "flag": "🇦🇺"},
    {"code": "IN", "name": "India", "flag": "🇮🇳"},
    {"code": "JP", "name": "Japan", "flag": "🇯🇵"},
    # ... full list of all ISO 3166 countries
]

CITIES_DATABASE = {
    "US": [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
        {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
        # ... more US cities
    ],
    "GB": [
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        # ... more UK cities
    ],
}

LOCATION_UI = """
┌─ ENROLLMENT FORM ────────────────────┐
│ Name: [John Doe]                     │
│ Category: [Criminal ▼]               │
│ DOB: [1985-03-15]                    │
│ Gender: [Male ▼]                     │
│                                      │
│ Country: [🇺🇸 United States ▼]       │
│ City: [New Y...]                     │
│         └─ New York                  │
│         └─ New Delhi                 │
│         └─ Newark                    │
│         └─ New Haven                 │
│                                      │
│ [Create Person & Enroll]             │
└──────────────────────────────────────┘
"""


# ============================================================================
# ISSUE 4: DATE OF BIRTH VALIDATION & PICKER
# ============================================================================
"""
PROBLEM: DOB is free-text, allows invalid dates ("2099-13-45").

SOLUTION:
  1. Date input field with HTML5 picker
  2. Server-side validation (not future, not extreme past)
  3. Calculate age for optional display
  4. Store in ISO 8601 format (DD-MM-YYYY)

CONSTRAINTS:
  - Age >= 0 (not born in future)
  - Age <= 150 (not from 1800s)
  - Optional field (allow empty)

FRONTEND:
  <input type="date" id="dob" name="dob" max="2026-04-14">
  
BACKEND VALIDATION:
  - Check date is valid ISO format
  - Check age constraints
  - Return error if invalid
"""

DOB_VALIDATION = {
    "min_year": 1876,  # Max 150 years ago
    "max_date": "today",  # Not in future
    "allow_null": True,
}


# ============================================================================
# IMPLEMENTATION ROADMAP
# ============================================================================
"""
PRIORITY 1 (Day 1-2): Quick Wins
  [ ] Add countries dropdown + basic city field
  [ ] Add date picker to DOB field
  [ ] Add profile_photo_path to PersonRecord
  [ ] Add /api/v1/persons/{id}/profile-photo endpoint

PRIORITY 2 (Day 2-3): Camera Capture
  [ ] Create camera capture UI component
  [ ] Add /api/v1/quality/assess endpoint (real-time feedback)
  [ ] Add /api/v1/detection/quick endpoint (face detection)
  [ ] Integrate into enrollment page

PRIORITY 3 (Day 3-4): Location Intelligence  
  [ ] Add /api/v1/geo/countries endpoint
  [ ] Add /api/v1/geo/cities endpoint
  [ ] Add /api/v1/geo/validate endpoint
  [ ] City autocomplete with real-time suggestions

PRIORITY 4 (Day 4-5): Profile Photo Auto-Capture
  [ ] Hook into detection pipeline
  [ ] Auto-save best match image (confidence > 60%)
  [ ] Display profile photos in UI
  [ ] Add profile update triggers
"""


# ============================================================================
# FILE CHANGES SUMMARY
# ============================================================================
"""
BACKEND CHANGES:
  1. src/trace_aml/core/models.py
     - Add profile_photo_path, profile_photo_confidence to PersonRecord
  
  2. src/trace_aml/store/vector_store.py
     - Add profile_photo columns to persons table schema
     - Add update_person_profile_photo() method
  
  3. src/trace_aml/service/person_api.py
     - POST /api/v1/persons/{id}/profile-photo (update profile pic)
     - GET /api/v1/geo/countries (country list)
     - GET /api/v1/geo/cities (city list + autocomplete)
  
  4. src/trace_aml/service/quality_api.py (new file)
     - POST /api/v1/quality/assess (real-time quality feedback)
     - POST /api/v1/detection/quick (face detection for capture)
  
  5. src/trace_aml/pipeline/session.py
     - Add profile_photo_capture logic to recognition loop
     - Track best_confidence per person
     - Auto-update profile when confidence > threshold

FRONTEND CHANGES:
  1. src/frontend/shared/geo.json
     - Full ISO 3166 countries + major cities database
  
  2. src/frontend/enrollment/enrollment.js
     - Add calendar picker for DOB
     - Add country dropdown
     - Add city autocomplete
     - Add camera capture component
     - Handle captured images in upload flow
  
  3. src/frontend/enrollment/index.html
     - Add camera capture section
     - Add date picker input
     - Add location fields
  
  4. src/frontend/shared/styles
     - Camera preview styling
     - Quality feedback indicators
     - Autocomplete dropdown styling
  
  5. src/frontend/live_ops/live_ops.js
     - Display profile photos in person cards
"""

print(__doc__)
