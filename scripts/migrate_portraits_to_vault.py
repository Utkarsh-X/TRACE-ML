#!/usr/bin/env python3
"""Migrate existing data/portraits/*.jpg into the DataVault encrypted store.

This script is idempotent: re-running it is safe — it skips portraits
that are already in the vault index.  The originals are kept until
you pass --delete-originals.

Usage::

    python scripts/migrate_portraits_to_vault.py
    python scripts/migrate_portraits_to_vault.py --delete-originals

Prerequisites:
    - TRACE_VAULT_KEY must be set in .env (or environment)
    - Run from the project root (where data/ lives)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing trace_aml from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate portraits to DataVault")
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete the original data/portraits/*.jpg files after migration",
    )
    parser.add_argument(
        "--portraits-dir",
        default="data/portraits",
        help="Source portraits directory (default: data/portraits)",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=".env", override=False)

    from trace_aml.core.config import Settings, load_settings
    settings = load_settings()

    from trace_aml.store.data_vault import DataVault
    vault = DataVault(settings)

    portraits_dir = Path(args.portraits_dir)
    if not portraits_dir.exists():
        print(f"[migrate] No portraits directory found at {portraits_dir} — nothing to do.")
        return

    jpg_files = sorted(portraits_dir.glob("*.jpg")) + sorted(portraits_dir.glob("*.jpeg"))
    meta_files = set(portraits_dir.glob("*.meta.json"))

    if not jpg_files:
        print(f"[migrate] No .jpg files found in {portraits_dir} — nothing to do.")
        return

    print(f"[migrate] Found {len(jpg_files)} portrait(s) to migrate.")
    migrated = 0
    skipped  = 0

    for jpg_path in jpg_files:
        # Derive entity_id from filename stem (e.g. PRC001.jpg → PRC001)
        entity_id = jpg_path.stem

        # Skip if already in vault
        if vault.has_portrait(entity_id):
            print(f"  SKIP  {entity_id} (already in vault)")
            skipped += 1
            continue

        # Read score from meta.json sidecar if available
        score = 0.0
        meta_path = jpg_path.with_suffix(".meta.json")
        if meta_path.exists():
            import json
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                score = float(meta.get("score", 0.0))
            except Exception:
                pass

        jpeg_bytes = jpg_path.read_bytes()
        vault.put_portrait(entity_id, jpeg_bytes, score)
        print(f"  OK    {entity_id} → vault (score={score:.3f})")
        migrated += 1

    print()
    print(f"[migrate] Done: {migrated} migrated, {skipped} already in vault.")

    if args.delete_originals and migrated > 0:
        print(f"[migrate] Deleting originals from {portraits_dir} ...")
        for jpg_path in jpg_files:
            entity_id = jpg_path.stem
            if vault.has_portrait(entity_id):
                jpg_path.unlink(missing_ok=True)
                meta_path = jpg_path.with_suffix(".meta.json")
                meta_path.unlink(missing_ok=True)
        print("[migrate] Originals deleted.")

    vault_stats = vault.stats()
    print(f"[migrate] Vault now contains {vault_stats['portraits']} portrait(s).")


if __name__ == "__main__":
    main()
