#!/usr/bin/env python3
"""
Deploy script for AuditAnchor.sol on local Anvil (or any EVM node).

Usage:
    py scripts/deploy_audit_anchor.py
    or
    py -m scripts.deploy_audit_anchor

Environment Variables:
    TRUSTCV_ETH_RPC_URL          (default: http://127.0.0.1:8545)
    TRUSTCV_ETH_PRIVATE_KEY      (default: standard Anvil account 0 dev key)
    TRUSTCV_ETH_CHAIN_ID         (default: 31337)
"""

import json
import os
import sys
from pathlib import Path

# Standard Anvil Account #0 public development key (NOT for production!)
ANVIL_DEFAULT_DEV_KEY = (
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
)
DEFAULT_RPC_URL = "http://127.0.0.1:8545"
DEFAULT_CHAIN_ID = 31337

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_JSON = ROOT / "contracts" / "AuditAnchor.json"
CONTRACT_SOL = ROOT / "contracts" / "AuditAnchor.sol"


def get_compiled_contract():
    """Load precompiled artifact or compile from source using solcx."""
    if CONTRACT_JSON.exists():
        try:
            with open(CONTRACT_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "abi" in data and "bin" in data:
                    return data["abi"], data["bin"]
        except Exception:
            pass

    # Fallback to dynamic compilation
    try:
        import solcx
        solcx.set_solc_version("0.8.20")
        compiled = solcx.compile_files([str(CONTRACT_SOL)], output_values=["abi", "bin"])
        key = "contracts/AuditAnchor.sol:AuditAnchor"
        if key not in compiled:
            key = [k for k in compiled.keys() if "AuditAnchor" in k][0]
        abi = compiled[key]["abi"]
        bytecode = compiled[key]["bin"]
        with open(CONTRACT_JSON, "w", encoding="utf-8") as f:
            json.dump({"abi": abi, "bin": bytecode}, f, indent=2)
        return abi, bytecode
    except Exception as exc:
        raise RuntimeError(
            f"Could not load or compile AuditAnchor.sol: {exc}"
        ) from exc


def deploy():
    print("=" * 60)
    print("TrustCV Local AuditAnchor Deployment")
    print("=" * 60)

    try:
        from web3 import Web3
        from eth_account import Account
    except ImportError:
        print("[ERROR] web3 or eth_account is not installed.")
        print("Install via: py -m pip install web3 eth-account")
        sys.exit(1)

    rpc_url = os.environ.get("TRUSTCV_ETH_RPC_URL", DEFAULT_RPC_URL).strip()
    private_key = os.environ.get("TRUSTCV_ETH_PRIVATE_KEY", ANVIL_DEFAULT_DEV_KEY).strip()
    chain_id = int(os.environ.get("TRUSTCV_ETH_CHAIN_ID", str(DEFAULT_CHAIN_ID)))

    print(f"Connecting to RPC endpoint: {rpc_url} ...")
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 5}))

    if not w3.is_connected():
        print(f"[OFFLINE] Cannot connect to Ethereum node at {rpc_url}.")
        print("Please start Anvil in a separate terminal:")
        print("    anvil")
        print("or specify a running RPC endpoint using TRUSTCV_ETH_RPC_URL.")
        sys.exit(2)

    actual_chain_id = w3.eth.chain_id
    print(f"[CONNECTED] Connected to chain ID: {actual_chain_id}")

    account = Account.from_key(private_key)
    print(f"Deployer account: {account.address}")
    balance = w3.eth.get_balance(account.address)
    print(f"Deployer balance: {w3.from_wei(balance, 'ether')} ETH")

    if balance == 0:
        print("[ERROR] Deployer account has 0 ETH balance. Ensure local Anvil is running.")
        sys.exit(3)

    abi, bytecode = get_compiled_contract()
    if not bytecode.startswith("0x"):
        bytecode = "0x" + bytecode

    print("Deploying AuditAnchor contract...")
    contract_factory = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(account.address)

    construct_tx = contract_factory.constructor().build_transaction({
        "from": account.address,
        "nonce": nonce,
        "gas": 1500000,
        "gasPrice": w3.eth.gas_price,
        "chainId": actual_chain_id,
    })

    signed_tx = account.sign_transaction(construct_tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"Deployment transaction sent: {tx_hash.hex()}")
    print("Waiting for transaction receipt...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    if receipt.status != 1:
        print("[FAILED] Contract deployment transaction reverted on-chain.")
        sys.exit(4)

    deployed_address = receipt.contractAddress
    print("\n" + "=" * 60)
    print("[SUCCESS] AuditAnchor deployed successfully!")
    print("=" * 60)
    print(f"Contract Address : {deployed_address}")
    print(f"Transaction Hash : {tx_hash.hex()}")
    print(f"Block Number     : {receipt.blockNumber}")
    print(f"Chain ID         : {actual_chain_id}")
    print("\nTo configure TrustCV for this deployment, set:")
    print(f"set TRUSTCV_ETH_RPC_URL={rpc_url}")
    print(f"set TRUSTCV_ETH_CONTRACT_ADDRESS={deployed_address}")
    print(f"set TRUSTCV_ETH_PRIVATE_KEY={private_key}")
    print(f"set TRUSTCV_ETH_CHAIN_ID={actual_chain_id}")
    print("=" * 60)

    # Also update .env file if present
    env_file = ROOT / ".env"
    env_lines = [
        f"TRUSTCV_ETH_RPC_URL={rpc_url}\n",
        f"TRUSTCV_ETH_CONTRACT_ADDRESS={deployed_address}\n",
        f"TRUSTCV_ETH_PRIVATE_KEY={private_key}\n",
        f"TRUSTCV_ETH_CHAIN_ID={actual_chain_id}\n",
    ]
    try:
        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(env_lines)
        print(f"Updated local configuration in {env_file}")
    except Exception as e:
        print(f"Note: Could not write to .env: {e}")

    return deployed_address


if __name__ == "__main__":
    deploy()
