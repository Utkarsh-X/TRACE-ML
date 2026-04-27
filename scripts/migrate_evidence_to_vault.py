#!/usr/bin/env python3
"""Migrate existing data/screenshots/DET-*.jpg into the DataVault evidence store.

This script is idempotent — re-running skips screenshots already registered
in the vault evidence index.  Originals are kept unless --delete-originals.

The detection_id is derived from the screenshot filename stem (e.g.
``DET-20260419T082742Z-abc123.jpg`` → ``DET-20260419T082742Z-abc123``).

The entity_id is set to 'migrated' for all legacy files since the old
screenshots did not embed entity information in their names.

Usage::

    python scripts/migrate_evidence_to_vault.py
    python scripts/migrate_evidence_to_vault.py --delete-originals
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate evidence screenshots to DataVault")
    parser.add_argument("--delete-originals", action="store_true")
    parser.add_argument("--screenshots-dir", default="data/screenshots")
    args = parser.parse_args()

    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=".env", override=False)

    from trace_aml.core.config import load_settings
    settings = load_settings()

    from trace_aml.store.data_vault import DataVault
    vault = DataVault(settings)

    screenshots_dir = Path(args.screenshots_dir)
    if not screenshots_dir.exists():
        print(f"[migrate] No screenshots directory at {screenshots_dir} — nothing to do.")
        return

    jpg_files = sorted(screenshots_dir.glob("*.jpg")) + sorted(screenshots_dir.glob("*.jpeg"))
    if not jpg_files:
        print("[migrate] No screenshot files found — nothing to do.")
        return

    print(f"[migrate] Found {len(jpg_files)} screenshot(s) to migrate.")
    migrated = 0
    skipped  = 0

    for jpg_path in jpg_files:
        detection_id = jpg_path.stem
        if vault.has_evidence(detection_id):
            skipped += 1
            continue

        jpeg_bytes = jpg_path.read_bytes()
        vault.put_evidence(
            detection_id=detection_id,
            entity_id="migrated",
            jpeg_bytes=jpeg_bytes,
        )
        print(f"  OK    {detection_id}")
        migrated += 1

    print(f"\n[migrate] Done: {migrated} migrated, {skipped} already in vault.")

    if args.delete_originals and migrated > 0:
        print("[migrate] Deleting originals ...")
        for jpg_path in jpg_files:
            if vault.has_evidence(jpg_path.stem):
                jpg_path.unlink(missing_ok=True)
        print("[migrate] Originals deleted.")

    vault_stats = vault.stats()
    print(f"[migrate] Vault evidence count: {vault_stats['evidence']}")


if __name__ == "__main__":
    main()
