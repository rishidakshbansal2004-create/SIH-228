#!/usr/bin/env python3
"""
TrustCV Model Signing & Verification CLI

Implements Ed25519 asymmetric cryptographic signing for PyTorch (.pt) and ONNX models.
Provides detached signature generation (.sig), signed manifest envelopes (.manifest.json),
and offline verification against a trusted public key or ledger.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

DEFAULT_KEYS_DIR = Path(__file__).resolve().parent / "keys"
_env_priv = os.environ.get("TRUSTCV_PRIVATE_KEY_PATH") or os.environ.get("TRUSTCV_MODEL_PRIVATE_KEY_PATH")
DEFAULT_PRIVATE_KEY = Path(_env_priv) if _env_priv else (DEFAULT_KEYS_DIR / "vendor_private_key.pem")
_env_pub = os.environ.get("TRUSTCV_PUBLIC_KEY_PATH") or os.environ.get("TRUSTCV_MODEL_PUBLIC_KEY_PATH")
DEFAULT_PUBLIC_KEY = Path(_env_pub) if _env_pub else (DEFAULT_KEYS_DIR / "vendor_public_key.pem")
REGISTRY_PATH = Path(__file__).resolve().parent / "trusted_registry.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_sha256(file_path: str | Path) -> str:
    """Compute streaming SHA-256 hex digest of a model file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_keypair(
    private_key_path: Path = DEFAULT_PRIVATE_KEY,
    public_key_path: Path = DEFAULT_PUBLIC_KEY,
    force: bool = False,
) -> tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey, str]:
    """Generate Ed25519 private/public keypair and save to disk."""
    if private_key_path.exists() and not force:
        print(f"[!] Private key already exists at {private_key_path}. Use --force to regenerate.")
        with open(private_key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
        with open(public_key_path, "rb") as f:
            public_key = serialization.load_pem_public_key(f.read())
        raw_pub = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return private_key, public_key, base64.b64encode(raw_pub).decode("ascii")

    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    # Save PEM private key (restrict permissions to owner)
    with open(private_key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    os.chmod(private_key_path, 0o600)

    # Save PEM public key
    with open(public_key_path, "wb") as f:
        f.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

    raw_pub = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    pub_b64 = base64.b64encode(raw_pub).decode("ascii")
    print(f"[+] Successfully generated Ed25519 keypair:")
    print(f"    Private Key: {private_key_path}")
    print(f"    Public Key:  {public_key_path}")
    print(f"    Public Key (B64): {pub_b64}")
    return private_key, public_key, pub_b64


def load_private_key(private_key_path: Path = DEFAULT_PRIVATE_KEY) -> ed25519.Ed25519PrivateKey:
    env_key = os.environ.get("TRUSTCV_PRIVATE_KEY") or os.environ.get("TRUSTCV_MODEL_PRIVATE_KEY")
    if env_key:
        return serialization.load_pem_private_key(env_key.strip().encode("utf-8"), password=None)
    if not private_key_path.exists():
        raise FileNotFoundError(
            f"Private key not found at {private_key_path}. "
            "Set TRUSTCV_PRIVATE_KEY / TRUSTCV_PRIVATE_KEY_PATH, or run keygen."
        )
    with open(private_key_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def load_public_key(public_key_path: Path = DEFAULT_PUBLIC_KEY) -> ed25519.Ed25519PublicKey:
    if not public_key_path.exists():
        raise FileNotFoundError(f"Public key not found at {public_key_path}. Run --generate-keys first.")
    with open(public_key_path, "rb") as f:
        return serialization.load_pem_public_key(f.read())


def sign_model_file(
    model_path: str | Path,
    private_key_path: Path = DEFAULT_PRIVATE_KEY,
    key_id: str = "vendor-key-v2-rotated",
    update_ledger: bool = True,
) -> dict:
    """Sign a model file with Ed25519 and produce detached .sig and .manifest.json files."""
    model_file = Path(model_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    private_key = load_private_key(private_key_path)
    public_key = private_key.public_key()
    raw_pub = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    pub_b64 = base64.b64encode(raw_pub).decode("ascii")

    print(f"[*] Computing SHA-256 for '{model_file.name}'...")
    digest = compute_sha256(model_file)
    print(f"    SHA-256: {digest}")

    # Sign the SHA-256 digest bytes
    signature_bytes = private_key.sign(digest.encode("utf-8"))
    sig_b64 = base64.b64encode(signature_bytes).decode("ascii")

    # 1. Write detached signature file (.sig)
    sig_path = model_file.with_name(f"{model_file.name}.sig")
    with open(sig_path, "w", encoding="utf-8") as f:
        f.write(sig_b64.strip() + "\n")
    print(f"[+] Detached signature saved: {sig_path}")

    # 2. Write cryptographic manifest (.manifest.json)
    manifest = {
        "format": "trustcv-model-manifest-v1",
        "model_name": model_file.name,
        "model_path": str(model_file),
        "file_size_bytes": model_file.stat().st_size,
        "sha256": digest,
        "artifact_digest": f"sha256:{digest}",
        "signing_key_id": key_id,
        "signature_algorithm": "ed25519",
        "public_key": pub_b64,
        "signature": sig_b64,
        "signed_at": utc_now_iso(),
        "signer": "TrustCV Authorized Release Authority",
    }
    manifest_path = model_file.with_name(f"{model_file.name}.manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Signed manifest envelope saved: {manifest_path}")

    # 3. Update offline trusted registry if requested
    if update_ledger and REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)
            models = registry.setdefault("models", {})
            if digest in models:
                models[digest]["signed"] = True
                models[digest]["signature"] = sig_b64
                models[digest]["public_key"] = pub_b64
                models[digest]["signing_key_id"] = key_id
                models[digest]["signature_algorithm"] = "ed25519"
                models[digest]["signed_at"] = manifest["signed_at"]
                with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
                    json.dump(registry, f, indent=2)
                print(f"[+] Enrolled signature into trusted registry: {REGISTRY_PATH}")
        except Exception as e:
            print(f"[!] Warning: Could not update registry: {e}")

    return {
        "status": "SIGNED",
        "model_name": model_file.name,
        "sha256": digest,
        "signature": sig_b64,
        "public_key": pub_b64,
        "key_id": key_id,
        "sig_file": str(sig_path),
        "manifest_file": str(manifest_path),
    }


def verify_model_file(
    model_path: str | Path,
    signature_path: str | Path | None = None,
    public_key_path: Path = DEFAULT_PUBLIC_KEY,
    raw_public_key_b64: str | None = None,
) -> bool:
    """Verify a model file against its digital signature and public key."""
    model_file = Path(model_path).resolve()
    if not model_file.exists():
        print(f"[-] Error: Model file not found: {model_file}")
        return False

    # Find signature: supplied path, or .sig file, or .manifest.json, or registry
    sig_b64 = None
    if signature_path and Path(signature_path).exists():
        with open(signature_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("{"):
                try:
                    data = json.loads(content)
                    sig_b64 = data.get("signature")
                except Exception:
                    pass
            if not sig_b64:
                sig_b64 = content
    else:
        sig_file = model_file.with_name(f"{model_file.name}.sig")
        manifest_file = model_file.with_name(f"{model_file.name}.manifest.json")
        if sig_file.exists():
            with open(sig_file, "r", encoding="utf-8") as f:
                sig_b64 = f.read().strip()
        elif manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                sig_b64 = json.load(f).get("signature")

    # Load public key
    if raw_public_key_b64:
        raw_bytes = base64.b64decode(raw_public_key_b64)
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(raw_bytes)
    elif public_key_path.exists():
        public_key = load_public_key(public_key_path)
    else:
        print(f"[-] Error: No public key found at {public_key_path} and none provided.")
        return False

    if not sig_b64:
        print(f"[-] Error: No signature found for {model_file.name}")
        return False

    print(f"[*] Recomputing SHA-256 for '{model_file.name}'...")
    digest = compute_sha256(model_file)
    print(f"    Candidate SHA-256: {digest}")

    try:
        sig_bytes = base64.b64decode(sig_b64)
        public_key.verify(sig_bytes, digest.encode("utf-8"))
        print("\n" + "=" * 60)
        print(" [OK] CRYPTOGRAPHIC SIGNATURE VERIFIED SUCCESSFULLY!")
        print(f"     Model:        {model_file.name}")
        print(f"     Digest:       sha256:{digest}")
        print(f"     Algorithm:    Ed25519")
        print(f"     Integrity:    100% Intact (0 bit divergence)")
        print(f"     Authenticity: Certified Release Authority")
        print("=" * 60 + "\n")
        return True
    except InvalidSignature:
        print("\n" + "!" * 60)
        print(" [X] CRITICAL SECURITY ALERT: INVALID SIGNATURE!")
        print(f"     The model '{model_file.name}' does NOT match this signature.")
        print("     Possible causes: Unauthorized modification, Trojan insertion, or forged key.")
        print("!" * 60 + "\n")
        return False
    except Exception as e:
        print(f"[-] Verification error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="TrustCV Model Digital Signature Utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Keygen
    keygen_p = subparsers.add_parser("keygen", help="Generate vendor Ed25519 keypair")
    keygen_p.add_argument("--force", action="store_true", help="Overwrite existing keypair")

    # Sign
    sign_p = subparsers.add_parser("sign", help="Sign a model file with Ed25519")
    sign_p.add_argument("model", help="Path to model file (e.g., best.pt)")
    sign_p.add_argument("--key-id", default="vendor-key-v2-rotated", help="Key identifier")

    # Verify
    verify_p = subparsers.add_parser("verify", help="Verify digital signature of a model")
    verify_p.add_argument("model", help="Path to model file (e.g., best.pt)")
    verify_p.add_argument("--sig", default=None, help="Optional explicit signature path")

    args = parser.parse_args()

    if args.command == "keygen":
        generate_keypair(force=args.force)
    elif args.command == "sign":
        # Ensure keypair exists first
        if not DEFAULT_PRIVATE_KEY.exists():
            print("[*] No existing keypair found. Generating new keypair first...")
            generate_keypair()
        sign_model_file(args.model, key_id=args.key_id)
    elif args.command == "verify":
        success = verify_model_file(args.model, signature_path=args.sig)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
