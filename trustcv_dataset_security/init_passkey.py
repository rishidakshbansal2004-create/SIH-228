"""Initialize local TrustCV contributor access and dataset trust credentials."""

import getpass
import hashlib
import json
from pathlib import Path

from trustcv_mirad_dataset_security import ensure_trust_anchor


def main():
    value = getpass.getpass("Create TrustCV contributor passkey: ").strip()
    if len(value) < 8:
        raise SystemExit("Passkey must be at least 8 characters.")

    confirm = getpass.getpass("Confirm passkey: ").strip()
    if value != confirm:
        raise SystemExit("Passkeys do not match.")

    target = Path.home() / "TrustCV_access_config.json"
    payload = {
        "passkey_sha256": hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest(),
        "purpose": "TrustCV local contributor passkey gate",
    }

    target.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    anchor = ensure_trust_anchor()

    print(f"Contributor passkey configuration created: {target}")
    print("Keep this file private. Do not commit it.")
    print()
    print("Dataset trust anchor initialized:")
    print(f"  Trust anchor: {anchor['trust_anchor_id']}")
    print(f"  Key ID:       {anchor['key_id']}")
    print("  Algorithm:    Ed25519")
    print("  Private key:  stored outside the project under your user profile.")


if __name__ == "__main__":
    main()
