"""
Ethereum Bridge Adapter

Bridges quantum results and NFTs to Ethereum network.
Interacts with Solidity smart contracts for verification and minting.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from web3 import Web3
    from web3.contract import Contract
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logger.warning("web3.py not available - Ethereum bridge disabled")

from .bridge_types import (
    QuantumResultBridgeData,
    NFTBridgeData,
    BridgeProof
)


class EthereumBridge:
    """
    Ethereum bridge for quantum results and NFTs.
    
    Interacts with deployed smart contracts:
    - QuantumResultBridge.sol: Verifies and stores quantum results
    - QuantumAchievementNFT.sol: Mints wrapped NFTs on Ethereum
    """
    
    # Contract ABIs (simplified - would load from JSON)
    RESULT_BRIDGE_ABI = [
        {
            "name": "bridgeQuantumResult",
            "type": "function",
            "inputs": [
                {"name": "jobId", "type": "string"},
                {"name": "cid", "type": "string"},
                {"name": "circuitHash", "type": "bytes32"},
                {"name": "backend", "type": "string"},
                {"name": "numQubits", "type": "uint256"},
                {"name": "circuitDepth", "type": "uint256"},
                {"name": "resultHash", "type": "bytes32"},
                {"name": "proof", "type": "bytes"}
            ],
            "outputs": [{"name": "success", "type": "bool"}]
        },
        {
            "name": "verifyQuantumResult",
            "type": "function",
            "inputs": [
                {"name": "jobId", "type": "string"},
                {"name": "proof", "type": "bytes"}
            ],
            "outputs": [{"name": "valid", "type": "bool"}]
        }
    ]
    
    NFT_BRIDGE_ABI = [
        {
            "name": "mintBridgedNFT",
            "type": "function",
            "inputs": [
                {"name": "to", "type": "address"},
                {"name": "nftId", "type": "string"},
                {"name": "achievementType", "type": "string"},
                {"name": "metadataURI", "type": "string"}
            ],
            "outputs": [{"name": "tokenId", "type": "uint256"}]
        }
    ]
    
    def __init__(
        self,
        rpc_url: str = "http://localhost:8545",
        chain_id: int = 1,
        result_bridge_address: Optional[str] = None,
        nft_bridge_address: Optional[str] = None,
        private_key: Optional[str] = None
    ):
        """
        Initialize Ethereum bridge.
        
        Args:
            rpc_url: Ethereum RPC endpoint
            chain_id: Chain ID (1=mainnet, 5=goerli, 11155111=sepolia)
            result_bridge_address: Deployed QuantumResultBridge contract address
            nft_bridge_address: Deployed QuantumAchievementNFT contract address
            private_key: Private key for signing transactions
        """
        if not WEB3_AVAILABLE:
            raise RuntimeError("web3.py not installed. Install with: pip install web3")
        
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.chain_id = chain_id
        
        # Load account
        if private_key:
            self.account = Account.from_key(private_key)
        else:
            # Development account (DO NOT USE IN PRODUCTION)
            logger.warning("No private key provided - using development account")
            self.account = None
        
        # Load contracts
        self.result_bridge: Optional[Contract] = None
        self.nft_bridge: Optional[Contract] = None
        
        if result_bridge_address:
            self.result_bridge = self.w3.eth.contract(
                address=Web3.to_checksum_address(result_bridge_address),
                abi=self.RESULT_BRIDGE_ABI
            )
        
        if nft_bridge_address:
            self.nft_bridge = self.w3.eth.contract(
                address=Web3.to_checksum_address(nft_bridge_address),
                abi=self.NFT_BRIDGE_ABI
            )
        
        logger.info(f"Initialized Ethereum bridge (chain_id: {chain_id})")
    
    async def bridge_quantum_result(
        self,
        result_data: QuantumResultBridgeData,
        destination_account: str,
        proof: Optional[BridgeProof]  
    ) -> str:
        """
        Bridge quantum result to Ethereum.
        
        Calls QuantumResultBridge.bridgeQuantumResult() contract method.
        
        Args:
            result_data: Quantum result data
            destination_account: Ethereum address to receive result proof
            proof: Cryptographic proof
            
        Returns:
            Ethereum transaction hash
        """
        if not self.result_bridge:
            raise RuntimeError("Result bridge contract not configured")
        
        if not self.account:
            raise RuntimeError("No account configured for signing")
        
        logger.info(f"Bridging quantum result {result_data.job_id} to Ethereum")
        
        # Prepare proof bytes
        proof_bytes = proof.proof_data if proof else b''
        
        # Build transaction
        tx = self.result_bridge.functions.bridgeQuantumResult(
            result_data.job_id,
            result_data.cid,
            result_data.circuit_hash,
            result_data.backend,
            result_data.num_qubits,
            result_data.circuit_depth,
            result_data.result_hash,
            proof_bytes
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 500000,
            'gasPrice': self.w3.eth.gas_price,
            'chainId': self.chain_id
        })
        
        # Sign and send
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for confirmation
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        
        if receipt['status'] == 1:
            logger.info(f"✅ Quantum result bridged: {tx_hash.hex()}")
            return tx_hash.hex()
        else:
            raise RuntimeError(f"Transaction failed: {tx_hash.hex()}")
    
    async def bridge_nft(
        self,
        nft_data: NFTBridgeData,
        destination_account: str
    ) -> str:
        """
        Bridge achievement NFT to Ethereum.
        
        Mints wrapped NFT on Ethereum representing BelizeChain achievement.
        
        Args:
            nft_data: NFT metadata
            destination_account: Ethereum address to receive NFT
            
        Returns:
            Ethereum transaction hash
        """
        if not self.nft_bridge:
            raise RuntimeError("NFT bridge contract not configured")
        
        if not self.account:
            raise RuntimeError("No account configured for signing")
        
        logger.info(f"Bridging NFT {nft_data.nft_id} to Ethereum")
        
        # Build transaction
        tx = self.nft_bridge.functions.mintBridgedNFT(
            Web3.to_checksum_address(destination_account),
            nft_data.nft_id,
            nft_data.achievement_type,
            nft_data.metadata_uri
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 300000,
            'gasPrice': self.w3.eth.gas_price,
            'chainId': self.chain_id
        })
        
        # Sign and send
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for confirmation
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        
        if receipt['status'] == 1:
            logger.info(f"✅ NFT bridged: {tx_hash.hex()}")
            return tx_hash.hex()
        else:
            raise RuntimeError(f"Transaction failed: {tx_hash.hex()}")
    
    async def verify_proof(
        self,
        job_id: str,
        proof: BridgeProof
    ) -> bool:
        """
        Verify quantum result proof from Ethereum.
        
        Args:
            job_id: Job ID to verify
            proof: Proof from Ethereum
            
        Returns:
            True if proof is valid
        """
        if not self.result_bridge:
            logger.error("Result bridge contract not configured")
            return False
        
        try:
            # Call contract view function
            is_valid = self.result_bridge.functions.verifyQuantumResult(
                job_id,
                proof.proof_data
            ).call()
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Proof verification failed: {e}")
            return False
    
    def get_gas_estimate(
        self,
        operation: str,
        **kwargs
    ) -> int:
        """
        Estimate gas cost for bridge operation.
        
        Args:
            operation: 'bridge_result' or 'bridge_nft'
            **kwargs: Operation-specific parameters
            
        Returns:
            Estimated gas in wei
        """
        try:
            if operation == 'bridge_result':
                gas_estimate = 500000  # Typical for result bridging
            elif operation == 'bridge_nft':
                gas_estimate = 300000  # Typical for NFT minting
            else:
                gas_estimate = 200000  # Default
            
            gas_price = self.w3.eth.gas_price
            return gas_estimate * gas_price
            
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}")
            return 0
    
    def check_connection(self) -> bool:
        """Check if connected to Ethereum node."""
        try:
            return self.w3.is_connected()
        except:
            return False


# Convenience function for testing
async def test_ethereum_bridge():
    """Test Ethereum bridge connection."""
    bridge = EthereumBridge(
        rpc_url="http://localhost:8545",
        chain_id=1337  # Local testnet
    )
    
    if bridge.check_connection():
        print("✅ Connected to Ethereum")
        print(f"   Chain ID: {bridge.chain_id}")
        print(f"   Latest block: {bridge.w3.eth.block_number}")
        return True
    else:
        print("❌ Not connected to Ethereum")
        return False


if __name__ == "__main__":
    asyncio.run(test_ethereum_bridge())
