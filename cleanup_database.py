#!/usr/bin/env python3
"""
TRACE-AML Database Cleanup Script

This script completely clears all LanceDB tables and resets the system to a
fresh state. Use this before starting a new testing cycle.

Usage:
    python cleanup_database.py
"""

import shutil
from pathlib import Path
from trace_aml.core.config import load_settings
from trace_aml.core.logger import configure_logger
from loguru import logger


def cleanup_database():
    """Completely reset all database records and vectors."""
    
    print("╭────────────────────────────────────────────────╮")
    print("│ TRACE-AML Database Cleanup                      │")
    print("│ WARNING: This will delete ALL records            │")
    print("╰────────────────────────────────────────────────╯\n")
    
    # Load settings
    settings = load_settings()
    configure_logger(settings)
    
    # Directories to completely clear
    vectors_dir = Path(settings.store.vectors_dir)
    logs_dir = Path(settings.store.root) / "logs"
    exports_dir = Path(settings.store.exports_dir)
    screenshots_dir = Path(settings.store.screenshots_dir)
    person_images_dir = Path(settings.store.root) / "person_images"
    
    logger.info("Starting database cleanup...")
    logger.info(f"Vectors directory: {vectors_dir}")
    logger.info(f"Logs directory: {logs_dir}")
    logger.info(f"Exports directory: {exports_dir}")
    logger.info(f"Screenshots directory: {screenshots_dir}")
    logger.info(f"Person images directory: {person_images_dir}")
    
    # Confirm action
    response = input("\n⚠️  This will PERMANENTLY DELETE all database records. Continue? (yes/no): ").strip().lower()
    if response != "yes":
        print("❌ Cleanup cancelled.")
        return False
    
    # Clear vectors (LanceDB tables)
    if vectors_dir.exists():
        logger.info(f"Deleting vectors directory: {vectors_dir}")
        shutil.rmtree(vectors_dir)
        vectors_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Cleared {vectors_dir}")
    
    # Clear logs (handle locked files)
    if logs_dir.exists():
        logger.info(f"Deleting logs directory: {logs_dir}")
        try:
            shutil.rmtree(logs_dir)
        except PermissionError:
            # Log file might be locked, try to delete children individually
            for item in logs_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    else:
                        shutil.rmtree(item)
                except:
                    pass
        logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Cleared {logs_dir}")
    
    # Clear exports
    if exports_dir.exists():
        logger.info(f"Deleting exports directory: {exports_dir}")
        shutil.rmtree(exports_dir)
        exports_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Cleared {exports_dir}")
    
    # Clear screenshots
    if screenshots_dir.exists():
        logger.info(f"Deleting screenshots directory: {screenshots_dir}")
        shutil.rmtree(screenshots_dir)
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Cleared {screenshots_dir}")
    
    # Clear person images
    if person_images_dir.exists():
        logger.info(f"Deleting person_images directory: {person_images_dir}")
        shutil.rmtree(person_images_dir)
        person_images_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Cleared {person_images_dir}")
    
    print("\n╭────────────────────────────────────────────────╮")
    print("│ ✓ DATABASE CLEANUP COMPLETE                    │")
    print("│                                                │")
    print("│ System is now in fresh, clean state            │")
    print("│ Ready for new testing cycle                    │")
    print("╰────────────────────────────────────────────────╯\n")
    
    return True


if __name__ == "__main__":
    try:
        success = cleanup_database()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        exit(1)
