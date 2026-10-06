"""
TrustCV Ethereum Audit Anchor — real blockchain integration.

Provides three modes:
  1. OFFLINE / NOT CONFIGURED — no Ethereum dependency
  2. LOCAL EVM — Anvil / Hardhat / Ganache at 127.0.0.1:8545
  3. EXTERNAL RPC — any Ethereum-compatible endpoint

This module NEVER fakes a blockchain transaction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# AuditAnchor ABI — the minimal interface we need.
# Generated from AuditAnchor.sol; only the relevant function selectors.
AUDIT_ANCHOR_ABI = json.loads("""[
  {
    "inputs": [
      {"internalType": "bytes32", "name": "auditDigest", "type": "bytes32"},
      {"internalType": "uint64",  "name": "sequence",    "type": "uint64"},
      {"internalType": "bytes32", "name": "datasetDigest","type": "bytes32"}
    ],
    "name": "anchor",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {"internalType": "bytes32", "name": "auditDigest", "type": "bytes32"}
    ],
    "name": "getAnchor",
    "outputs": [
      {"internalType": "uint64",  "name": "sequence",     "type": "uint64"},
      {"internalType": "bytes32", "name": "datasetDigest", "type": "bytes32"},
      {"internalType": "address", "name": "submitter",     "type": "address"},
      {"internalType": "uint256", "name": "blockNumber",   "type": "uint256"},
      {"internalType": "uint256", "name": "timestamp",     "type": "uint256"}
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "anchorCount",
    "outputs": [
      {"internalType": "uint256", "name": "", "type": "uint256"}
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "anonymous": false,
    "inputs": [
      {"indexed": true,  "internalType": "bytes32", "name": "auditDigest",  "type": "bytes32"},
      {"indexed": false, "internalType": "uint64",  "name": "sequence",     "type": "uint64"},
      {"indexed": false, "internalType": "bytes32", "name": "datasetDigest","type": "bytes32"},
      {"indexed": true,  "internalType": "address", "name": "submitter",    "type": "address"},
      {"indexed": false, "internalType": "uint256", "name": "timestamp",    "type": "uint256"}
    ],
    "name": "AuditAnchored",
    "type": "event"
  }
]""")


def _load_env_fallback():
    """Load settings from .env file if not already set in os.environ."""
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"):
        if candidate.exists():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if k not in os.environ and v:
                            os.environ[k] = v
            except Exception:
                pass


@dataclass
class EthereumConfig:
    """Ethereum connection configuration — sourced from env vars or .env."""
    rpc_url: str = ""
    private_key: str = ""
    contract_address: str = ""
    chain_id: int = 0

    @classmethod
    def from_env(cls) -> "EthereumConfig":
        _load_env_fallback()
        return cls(
            rpc_url=os.environ.get("TRUSTCV_ETH_RPC_URL", "").strip(),
            private_key=os.environ.get("TRUSTCV_ETH_PRIVATE_KEY", "").strip(),
            contract_address=os.environ.get("TRUSTCV_ETH_CONTRACT_ADDRESS", "").strip(),
            chain_id=int(os.environ.get("TRUSTCV_ETH_CHAIN_ID", "0") or "0"),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.rpc_url and self.private_key and self.contract_address)


@dataclass
class AnchorResult:
    """Result of an anchor submission or lookup."""
    success: bool = False
    status: str = "NOT CONFIGURED"
    tx_hash: str = ""
    block_number: int = 0
    chain_id: int = 0
    contract_address: str = ""
    audit_digest: str = ""
    dataset_digest: str = ""
    sequence: int = 0
    timestamp: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "tx_hash": self.tx_hash,
            "block_number": self.block_number,
            "chain_id": self.chain_id,
            "contract_address": self.contract_address,
            "audit_digest": self.audit_digest,
            "dataset_digest": self.dataset_digest,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "error": self.error,
        }


@dataclass
class VerificationResult:
    """Result of verifying a local digest against an on-chain anchor."""
    status: str = "NOT CONFIGURED"
    local_digest: str = ""
    anchored_digest: str = ""
    local_dataset_digest: str = ""
    anchored_dataset_digest: str = ""
    match: bool = False
    audit_match: bool = False
    dataset_match: bool = False
    sequence: int = 0
    tx_hash: str = ""
    block_number: int = 0
    chain_id: int = 0
    contract_address: str = ""
    network: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "local_digest": self.local_digest,
            "anchored_digest": self.anchored_digest,
            "local_dataset_digest": self.local_dataset_digest,
            "anchored_dataset_digest": self.anchored_dataset_digest,
            "match": self.match,
            "audit_match": self.audit_match,
            "dataset_match": self.dataset_match,
            "sequence": self.sequence,
            "tx_hash": self.tx_hash,
            "block_number": self.block_number,
            "chain_id": self.chain_id,
            "contract_address": self.contract_address,
            "network": self.network,
            "error": self.error,
        }


def _check_web3() -> bool:
    """Check if web3 is importable."""
    try:
        import web3  # noqa: F401
        return True
    except ImportError:
        return False


def _hex_to_bytes32(hex_str: str) -> bytes:
    """Convert a hex digest (with or without 0x/sha256: prefix) to 32 bytes."""
    clean = hex_str.lower().strip()
    for prefix in ("sha256:", "0x"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
    if len(clean) != 64:
        raise ValueError(f"Expected 64 hex chars, got {len(clean)}")
    return bytes.fromhex(clean)


def get_ethereum_status(config: EthereumConfig | None = None) -> dict[str, Any]:
    """Get current Ethereum configuration status without making transactions."""
    if config is None:
        config = EthereumConfig.from_env()

    if not config.rpc_url:
        return {"status": "NOT CONFIGURED", "detail": "TRUSTCV_ETH_RPC_URL not set."}

    if not config.private_key:
        return {"status": "NOT CONFIGURED", "detail": "TRUSTCV_ETH_PRIVATE_KEY not set."}

    if not config.contract_address:
        return {"status": "NOT CONFIGURED", "detail": "TRUSTCV_ETH_CONTRACT_ADDRESS not set."}

    if not _check_web3():
        return {"status": "OFFLINE", "detail": "web3 library not installed."}

    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(config.rpc_url, request_kwargs={"timeout": 5}))
        if not w3.is_connected():
            return {"status": "OFFLINE", "detail": f"Cannot connect to {config.rpc_url}"}

        chain_id = w3.eth.chain_id
        block = w3.eth.block_number
        return {
            "status": "CONNECTED",
            "rpc_url": config.rpc_url,
            "chain_id": chain_id,
            "block_number": block,
            "contract": config.contract_address,
        }
    except Exception as exc:
        return {"status": "OFFLINE", "detail": str(exc)}


def anchor_checkpoint(
    *,
    audit_digest: str,
    sequence: int,
    dataset_digest: str,
    config: EthereumConfig | None = None,
) -> AnchorResult:
    """Submit an audit checkpoint anchor to the Ethereum contract.

    Returns an AnchorResult. NEVER fakes a successful transaction.
    """
    if config is None:
        config = EthereumConfig.from_env()

    if not config.is_configured:
        return AnchorResult(status="NOT CONFIGURED", error="Ethereum is not configured.")

    if not _check_web3():
        return AnchorResult(status="OFFLINE", error="web3 library not installed.")

    try:
        from web3 import Web3
        from eth_account import Account

        w3 = Web3(Web3.HTTPProvider(config.rpc_url, request_kwargs={"timeout": 15}))
        if not w3.is_connected():
            return AnchorResult(status="OFFLINE", error=f"Cannot connect to {config.rpc_url}")

        account = Account.from_key(config.private_key)
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(config.contract_address),
            abi=AUDIT_ANCHOR_ABI,
        )

        audit_bytes = _hex_to_bytes32(audit_digest)
        dataset_bytes = _hex_to_bytes32(dataset_digest)

        chain_id = config.chain_id or w3.eth.chain_id
        nonce = w3.eth.get_transaction_count(account.address)

        tx = contract.functions.anchor(
            audit_bytes,
            sequence,
            dataset_bytes,
        ).build_transaction({
            "from": account.address,
            "nonce": nonce,
            "chainId": chain_id,
            "gas": 200000,
            "gasPrice": w3.eth.gas_price,
        })

        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

        if receipt.status != 1:
            return AnchorResult(
                status="FAILED",
                tx_hash=tx_hash.hex(),
                error="Transaction reverted on-chain.",
                chain_id=chain_id,
                contract_address=config.contract_address,
            )

        return AnchorResult(
            success=True,
            status="ANCHORED",
            tx_hash=tx_hash.hex(),
            block_number=receipt.blockNumber,
            chain_id=chain_id,
            contract_address=config.contract_address,
            audit_digest=audit_digest,
            dataset_digest=dataset_digest,
            sequence=sequence,
        )

    except Exception as exc:
        return AnchorResult(status="ERROR", error=str(exc))


def verify_anchor(
    *,
    audit_digest: str,
    local_audit_hash: str | None = None,
    local_dataset_hash: str | None = None,
    config: EthereumConfig | None = None,
) -> VerificationResult:
    """Verify a local MIRAD audit digest and dataset digest against the on-chain anchor.

    Possible statuses: MATCH, MISMATCH, NOT CONFIGURED, OFFLINE,
    ANCHOR NOT FOUND, VERIFICATION ERROR
    """
    if config is None:
        config = EthereumConfig.from_env()

    if not config.is_configured:
        return VerificationResult(status="NOT CONFIGURED")

    if not _check_web3():
        return VerificationResult(status="OFFLINE", error="web3 library not installed.")

    try:
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(config.rpc_url, request_kwargs={"timeout": 10}))
        if not w3.is_connected():
            return VerificationResult(status="OFFLINE", error=f"Cannot connect to {config.rpc_url}")

        contract = w3.eth.contract(
            address=Web3.to_checksum_address(config.contract_address),
            abi=AUDIT_ANCHOR_ABI,
        )

        audit_bytes = _hex_to_bytes32(audit_digest)
        result = contract.functions.getAnchor(audit_bytes).call()
        # result: (sequence, datasetDigest, submitter, blockNumber, timestamp)
        sequence, dataset_digest_bytes, submitter, block_number, timestamp = result

        if submitter == "0x" + "0" * 40 and block_number == 0:
            return VerificationResult(
                status="ANCHOR NOT FOUND",
                local_digest=audit_digest,
                chain_id=w3.eth.chain_id,
                contract_address=config.contract_address,
            )

        anchored_audit_str = audit_bytes.hex()
        anchored_ds_str = dataset_digest_bytes.hex()
        chain_id = w3.eth.chain_id

        vr = VerificationResult(
            local_digest=local_audit_hash or audit_digest,
            anchored_digest=anchored_audit_str,
            local_dataset_digest=local_dataset_hash or "",
            anchored_dataset_digest=anchored_ds_str,
            sequence=sequence,
            block_number=block_number,
            chain_id=chain_id,
            contract_address=config.contract_address,
            network=_chain_name(chain_id),
        )

        # 1. Audit digest comparison
        audit_ok = True
        if local_audit_hash:
            local_clean = local_audit_hash.lower().replace("sha256:", "").replace("0x", "")
            anchor_clean = anchored_audit_str.lower().replace("sha256:", "").replace("0x", "")
            audit_ok = (local_clean == anchor_clean)
        vr.audit_match = audit_ok

        # 2. Dataset digest comparison
        ds_ok = True
        if local_dataset_hash:
            local_ds_clean = local_dataset_hash.lower().replace("sha256:", "").replace("0x", "")
            anchor_ds_clean = anchored_ds_str.lower().replace("sha256:", "").replace("0x", "")
            ds_ok = (local_ds_clean == anchor_ds_clean)
        vr.dataset_match = ds_ok

        if audit_ok and ds_ok:
            vr.status = "MATCH"
            vr.match = True
        else:
            vr.status = "MISMATCH"
            vr.match = False

        return vr

    except Exception as exc:
        return VerificationResult(status="VERIFICATION ERROR", error=str(exc))


def _chain_name(chain_id: int) -> str:
    names = {
        1: "Ethereum Mainnet",
        5: "Goerli Testnet",
        11155111: "Sepolia Testnet",
        31337: "Local (Anvil/Hardhat)",
        1337: "Local (Ganache)",
    }
    return names.get(chain_id, f"Chain {chain_id}")
