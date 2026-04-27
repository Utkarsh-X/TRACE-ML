#!/usr/bin/env python3
"""Migrate existing data/person_images/ into the DataVault enrollment store.

Reads every image from data/person_images/{person_id}/ and encrypts it
into the vault enrollment store.  Idempotent — duplicate images (same
SHA-256 content hash) are not re-stored.

Usage::

    python scripts/migrate_enrollment_to_vault.py
    python scripts/migrate_enrollment_to_vault.py --delete-originals
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate enrollment images to DataVault")
    parser.add_argument("--delete-originals", action="store_true")
    parser.add_argument("--person-images-dir", default="data/person_images")
    args = parser.parse_args()

    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=".env", override=False)

    from trace_aml.core.config import load_settings
    settings = load_settings()

    from trace_aml.store.data_vault import DataVault
    vault = DataVault(settings)

    person_images_dir = Path(args.person_images_dir)
    if not person_images_dir.exists():
        print(f"[migrate] No person_images directory at {person_images_dir} — nothing to do.")
        return

    total_migrated = 0
    persons_migrated = 0

    for person_dir in sorted(person_images_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        image_files = [
            f for f in sorted(person_dir.iterdir())
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if not image_files:
            continue

        migrated_for_person = 0
        for img_path in image_files:
            jpeg_bytes = img_path.read_bytes()
            vault.put_enrollment_image(person_id, jpeg_bytes)
            migrated_for_person += 1
            total_migrated += 1

        print(f"  OK  {person_id}: {migrated_for_person} image(s)")
        persons_migrated += 1

    print(f"\n[migrate] Done: {persons_migrated} person(s), {total_migrated} image(s) migrated.")

    if args.delete_originals and total_migrated > 0:
        print("[migrate] Deleting original person_images/ directories ...")
        import shutil
        for person_dir in sorted(person_images_dir.iterdir()):
            if person_dir.is_dir():
                shutil.rmtree(person_dir, ignore_errors=True)
        print("[migrate] Done.")

    vault_stats = vault.stats()
    print(f"[migrate] Vault enrollment: {vault_stats['persons_enrolled']} person(s), "
          f"{vault_stats['enrollment']} image blob(s).")


if __name__ == "__main__":
    main()
