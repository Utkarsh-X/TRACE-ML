#!/usr/bin/env python3
"""Store the vault key in the OS keychain (Windows Credential Manager / macOS Keychain).

This script is used ONCE when setting up the Electron packaged app.
In browser / dev mode, the key lives in .env — you do NOT need this script.

Usage::

    # Generate a key and store it immediately:
    python scripts/store_vault_key.py --generate

    # Store an existing key (e.g. migrating from .env):
    python scripts/store_vault_key.py --key <your-64-hex-char-key>

    # Verify the stored key:
    python scripts/store_vault_key.py --verify

The key is stored under service="trace-aml", username="vault_key" in the
platform's native secure credential store.

Why this is needed for Electron
--------------------------------
Packaged Electron apps cannot rely on .env files (they'd be inside the
read-only app bundle). The OS keychain is the correct secure storage for
secrets in a desktop app — it is per-user, protected by OS login credentials,
and never appears in plaintext on disk.
"""

from __future__ import annotations

import argparse
import os
import sys

_SERVICE = "trace-aml"
_USERNAME = "vault_key"


def _require_keyring():
    try:
        import keyring
        return keyring
    except ImportError:
        print("ERROR: 'keyring' is not installed.")
        print("Install it with:  pip install keyring")
        sys.exit(1)


def cmd_generate(args) -> None:
    kr = _require_keyring()
    key = os.urandom(32).hex()
    kr.set_password(_SERVICE, _USERNAME, key)
    print()
    print("=" * 68)
    print("  TRACE-AML Vault Key — stored in OS keychain")
    print("=" * 68)
    print()
    print(f"  Key (for backup — keep this SAFE):")
    print(f"  {key}")
    print()
    print("  The key has been stored in your OS credential manager.")
    print("  Back up the key above in a secure password manager.")
    print("  Without it, data/vault/ is permanently unreadable.")
    print()
    print("=" * 68)


def cmd_store(hex_key: str) -> None:
    kr = _require_keyring()
    key_bytes = bytes.fromhex(hex_key)
    if len(key_bytes) != 32:
        print(f"ERROR: Key must be 64 hex chars (32 bytes), got {len(key_bytes) * 2}")
        sys.exit(1)
    kr.set_password(_SERVICE, _USERNAME, hex_key)
    print(f"  OK: vault key stored in OS keychain ({_SERVICE}/{_USERNAME})")


def cmd_verify() -> None:
    kr = _require_keyring()
    stored = kr.get_password(_SERVICE, _USERNAME)
    if not stored:
        print("  NOT FOUND: no vault key in OS keychain.")
        print("  Run: python scripts/store_vault_key.py --generate")
        sys.exit(1)
    try:
        key_bytes = bytes.fromhex(stored)
        assert len(key_bytes) == 32
        print(f"  OK: vault key present in OS keychain ({len(key_bytes)*8}-bit).")
    except Exception:
        print("  ERROR: stored value is not a valid 32-byte hex key.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage vault key in OS keychain")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate",
        action="store_true",
        help="Generate a new random key and store it in the OS keychain",
    )
    group.add_argument(
        "--key",
        metavar="HEX",
        help="Store an existing 64-hex-char key in the OS keychain",
    )
    group.add_argument(
        "--verify",
        action="store_true",
        help="Verify a key is stored in the OS keychain",
    )
    args = parser.parse_args()

    if args.generate:
        cmd_generate(args)
    elif args.key:
        cmd_store(args.key)
    elif args.verify:
        cmd_verify()


if __name__ == "__main__":
    main()
