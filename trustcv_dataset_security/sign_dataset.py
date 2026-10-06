"""Create a trusted Ed25519 manifest for one exact dataset artifact.

The private signing key is stored outside the project under the user's profile.
Never commit it.
"""

import base64
import hashlib
import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trustcv_mirad_dataset_security import (
    canonicalize,
    ensure_trust_anchor,
    sha256_bytes,
)

KEY_FILE = Path.home() / ".trustcv_dataset_security" / "dataset_signing_private.pem"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python sign_dataset.py "DATASET_PATH" '
            '"SOURCE" "OWNER" "VERSION"'
        )
        raise SystemExit(2)

    dataset_path = Path(sys.argv[1]).resolve()
    source = sys.argv[2] if len(sys.argv) > 2 else "Synthetic Demo Source"
    owner = sys.argv[3] if len(sys.argv) > 3 else "Approved Demo Owner"
    version = sys.argv[4] if len(sys.argv) > 4 else "1.0.0"

    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    anchor = ensure_trust_anchor()
    if not KEY_FILE.exists():
        raise SystemExit("Dataset signing key is not configured. Run init_passkey.py once.")

    private = serialization.load_pem_private_key(KEY_FILE.read_bytes(), password=None)
    if not isinstance(private, Ed25519PrivateKey):
        raise SystemExit("Configured signing key is not Ed25519.")

    digest = sha256_file(dataset_path)

    # Keep the original TrustCV manifest fields for UI compatibility while
    # adding an explicit MIRAD-derived trust anchor and canonical signature.
    manifest = {
        "manifest_type": "TrustCV Dataset Manifest",
        "manifest_version": "1.0",
        "dataset_name": dataset_path.name,
        "expected_sha256": digest,
        "source": source,
        "vendor": owner,
        "contributor_id": owner,
        "version": version,
        "created_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "file_size_bytes": dataset_path.stat().st_size,
        "signature_algorithm": "Ed25519",
        "hash_algorithm": "SHA-256",
        "canonicalization_version": "1.0",
        "trust_anchor_id": anchor["trust_anchor_id"],
        "key_id": anchor["key_id"],
        "public_key": anchor["public_key"],
    }

    signature = private.sign(canonicalize(manifest))
    manifest["signature"] = base64.b64encode(signature).decode("ascii")

    out_dir = dataset_path.parent
    manifest_path = out_dir / "trustcv_dataset_manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Created: {manifest_path}")
    print(f"SHA-256: {digest}")
    print(f"Signature: Ed25519")
    print(f"Trust anchor: {anchor['trust_anchor_id']}")
    print(f"Key ID: {anchor['key_id']}")


if __name__ == "__main__":
    main()
