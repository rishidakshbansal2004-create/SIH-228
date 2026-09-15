import base64, hashlib, json, sys
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

if len(sys.argv)!=4:
    print("Usage: python sign_model.py MODEL_PATH VENDOR VERSION")
    raise SystemExit(1)

model=Path(sys.argv[1])
vendor=sys.argv[2]
version=sys.argv[3]

digest=hashlib.sha256(model.read_bytes()).hexdigest()
private=rsa.generate_private_key(public_exponent=65537,key_size=3072)
public=private.public_key()

signature=private.sign(
    digest.encode("utf-8"),
    padding.PSS(mgf=padding.MGF1(hashes.SHA256()),salt_length=padding.PSS.MAX_LENGTH),
    hashes.SHA256(),
)

manifest={
    "artifact":"model",
    "filename":model.name,
    "expected_sha256":digest,
    "vendor":vendor,
    "version":version,
    "provenance":"Locally approved demo artifact",
    "signature_algorithm":"RSA-PSS-SHA256",
    "signature":base64.b64encode(signature).decode("ascii"),
    "public_key_pem":public.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")
}

Path("trustcv_model_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
Path("trustcv_private_key.pem").write_bytes(
    private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()
    )
)
print("Created trustcv_model_manifest.json")
print("Created trustcv_private_key.pem (KEEP PRIVATE)")
print("SHA-256:",digest)
