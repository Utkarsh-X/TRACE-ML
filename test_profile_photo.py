#!/usr/bin/env python
"""Test script to manually update profile photo in database."""
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from trace_aml.core.config import load_settings
from trace_aml.store.vector_store import VectorStore
from trace_aml.core.models import PersonRecord

settings = load_settings()
store = VectorStore(settings)

# Get the current person
person = store.get_person("PRM001")
if not person:
    print("Person not found")
    exit(1)

# Update with profile photo path
profile_path = str(Path("data/person_images/PRM001/best_detection_1776312195_88.jpg").resolve())
now = datetime.now(timezone.utc).isoformat()

print(f"Setting profile_photo_path to: {profile_path}")
print(f"Setting confidence to: 0.88")

# Create PersonRecord with updated fields
person_record = PersonRecord(
    person_id=person["person_id"],
    name=person["name"],
    category=person.get("category", "unknown"),
    profile_photo_path=profile_path,
    profile_photo_confidence=0.88,
    best_detection_id="DET-TEST-001",
    best_detection_timestamp=now,
    created_at=person.get("created_at", now),
    updated_at=now,
    lifecycle_state=person.get("lifecycle_state", "active"),
    lifecycle_reason=person.get("lifecycle_reason", ""),
    enrollment_score=float(person.get("enrollment_score", 0.0)),
    valid_embeddings=int(person.get("valid_embeddings", 0)),
    valid_images=int(person.get("valid_images", 0)),
    total_images=int(person.get("total_images", 0)),
)

store.add_or_update_person(person_record)
print("✓ Person record updated")

# Verify
updated = store.get_person("PRM001")
print(f"Verified profile_photo_path: {updated['profile_photo_path']}")
print(f"Verified confidence: {updated['profile_photo_confidence']}")
print(f"Verified best_detection_id: {updated['best_detection_id']}")
