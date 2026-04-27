#!/usr/bin/env python3
"""Generate a secure 32-byte vault key and print it as a 64-char hex string.

Usage::

    python scripts/generate_vault_key.py

Then copy the output into your .env file as::

    TRACE_VAULT_KEY=<output>

WARNING: Never share this key or commit it to version control.
Store it separately from the data/vault/ directory.
"""

import os
import sys


def main() -> None:
    key = os.urandom(32)
    hex_key = key.hex()
    print()
    print("=" * 68)
    print("  TRACE-AML DataVault Key Generator")
    print("=" * 68)
    print()
    print("  Generated key (64 hex chars = 32 bytes = 256-bit):")
    print()
    print(f"  TRACE_VAULT_KEY={hex_key}")
    print()
    print("  Instructions:")
    print("  1. Copy the line above into your .env file")
    print("  2. Keep .env backed up SEPARATELY from data/vault/")
    print("  3. Without this key, data/vault/ is unreadable — NEVER lose it")
    print()
    print("  [!] WARNING: This key is randomly generated and not stored anywhere.")
    print("     Run this script once per deployment. Do NOT regenerate unless you")
    print("     also re-encrypt all existing vault blobs.")
    print()
    print("=" * 68)
    print()


if __name__ == "__main__":
    main()
