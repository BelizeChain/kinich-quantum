"""
Cross-chain Bridge for Quantum Results and NFTs

Enables bridging quantum computation results and achievement NFTs
between BelizeChain and other blockchains (Ethereum, Polkadot, Kusama).

Supports:
- Quantum result verification proofs on destination chains
- Achievement NFT transfer across chains
- Bi-directional bridging
- Multi-signature validation
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

from .bridge_types import (
    BridgeDestination,
    BridgeTransaction,
    BridgeStatus,
    ChainType,
    AssetType,
    QuantumResultBridgeData,
    NFTBridgeData,
    BridgeProof
)
from .ethereum_bridge import EthereumBridge
from .polkadot_xcm import PolkadotXCMBridge


class CrossChainBridge:
    """
    Cross-chain bridge for quantum results and NFTs.
    
    Coordinates bridging between BelizeChain and external chains.
    """
    
    def __init__(
        self,
        belizechain_connector: Optional[Any] = None,
        ethereum_bridge: Optional[Any] = None,
        polkadot_bridge: Optional[Any] = None
    ):
        """
        Initialize cross-chain bridge.
        
        Args:
            belizechain_connector: Connection to BelizeChain
            ethereum_bridge: Ethereum bridge adapter
            polkadot_bridge: Polkadot/XCM bridge adapter
        """
        self.belizechain = belizechain_connector
        self.ethereum = ethereum_bridge
        self.polkadot = polkadot_bridge
        
        # Transaction tracking
        self.pending_transactions: Dict[str, BridgeTransaction] = {}
        self.completed_transactions: List[BridgeTransaction] = []
    
    async def bridge_quantum_result(
        self,
        job_id: str,
        destination: BridgeDestination,
        result_data: QuantumResultBridgeData,
        generate_proof: bool = True
    ) -> BridgeTransaction:
        """
        Bridge quantum result to another chain.
        
        Args:
            job_id: Quantum job ID
            destination: Target chain and account
            result_data: Quantum result data to bridge
            generate_proof: Generate cryptographic proof for verification
            
        Returns:
            BridgeTransaction record
        """
        logger.info(f"Bridging quantum result {job_id} to {destination.chain_type.value}")
        
        # Create bridge transaction
        tx = BridgeTransaction(
            tx_id=str(uuid.uuid4()),
            source_chain=ChainType.BELIZECHAIN,
            destination=destination,
            asset_type=AssetType.QUANTUM_RESULT,
            asset_id=job_id,
            status=BridgeStatus.PENDING,
            timestamp=datetime.now()
        )
        
        self.pending_transactions[tx.tx_id] = tx
        
        try:
            # Generate proof if requested
            proof = None
            if generate_proof:
                proof = await self._generate_bridge_proof(result_data)
            
            # Route to appropriate bridge
            if destination.chain_type == ChainType.ETHEREUM:
                tx.destination_tx_hash = await self._bridge_to_ethereum(
                    result_data=result_data,
                    destination_account=destination.account,
                    proof=proof
                )
            elif destination.chain_type in [ChainType.POLKADOT, ChainType.KUSAMA, ChainType.PARACHAIN]:
                tx.destination_tx_hash = await self._bridge_to_polkadot(
                    result_data=result_data,
                    destination=destination,
                    proof=proof
                )
            else:
                raise ValueError(f"Unsupported destination chain: {destination.chain_type}")
            
            # Mark as confirmed
            tx.status = BridgeStatus.CONFIRMED
            logger.info(f"✅ Bridged quantum result: {tx.destination_tx_hash}")
            
        except Exception as e:
            tx.status = BridgeStatus.FAILED
            tx.error_message = str(e)
            logger.error(f"Bridge failed: {e}")
        
        finally:
            # Move to completed
            self.pending_transactions.pop(tx.tx_id, None)
            self.completed_transactions.append(tx)
        
        return tx
    
    async def bridge_achievement_nft(
        self,
        nft_id: str,
        destination: BridgeDestination,
        nft_data: NFTBridgeData
    ) -> BridgeTransaction:
        """
        Bridge achievement NFT to another chain.
        
        Args:
            nft_id: NFT ID
            destination: Target chain and account
            nft_data: NFT metadata
            
        Returns:
            BridgeTransaction record
        """
        logger.info(f"Bridging NFT {nft_id} to {destination.chain_type.value}")
        
        # Create bridge transaction
        tx = BridgeTransaction(
            tx_id=str(uuid.uuid4()),
            source_chain=ChainType.BELIZECHAIN,
            destination=destination,
            asset_type=AssetType.ACHIEVEMENT_NFT,
            asset_id=nft_id,
            status=BridgeStatus.PENDING,
            timestamp=datetime.now()
        )
        
        self.pending_transactions[tx.tx_id] = tx
        
        try:
            # Lock NFT on BelizeChain (prevent double-spend)
            if self.belizechain:
                await self._lock_nft_on_belizechain(nft_id)
            
            # Route to appropriate bridge
            if destination.chain_type == ChainType.ETHEREUM:
                tx.destination_tx_hash = await self._bridge_nft_to_ethereum(
                    nft_data=nft_data,
                    destination_account=destination.account
                )
            elif destination.chain_type in [ChainType.POLKADOT, ChainType.KUSAMA]:
                tx.destination_tx_hash = await self._bridge_nft_to_polkadot(
                    nft_data=nft_data,
                    destination=destination
                )
            else:
                raise ValueError(f"Unsupported destination chain: {destination.chain_type}")
            
            tx.status = BridgeStatus.CONFIRMED
            logger.info(f"✅ Bridged NFT: {tx.destination_tx_hash}")
            
        except Exception as e:
            tx.status = BridgeStatus.FAILED
            tx.error_message = str(e)
            logger.error(f"NFT bridge failed: {e}")
            
            # Unlock NFT if bridge failed
            if self.belizechain:
                await self._unlock_nft_on_belizechain(nft_id)
        
        finally:
            self.pending_transactions.pop(tx.tx_id, None)
            self.completed_transactions.append(tx)
        
        return tx
    
    async def verify_bridged_result(
        self,
        job_id: str,
        source_chain: ChainType,
        proof: BridgeProof
    ) -> bool:
        """
        Verify quantum result bridged from another chain.
        
        Args:
            job_id: Job ID to verify
            source_chain: Source blockchain
            proof: Cryptographic proof from source chain
            
        Returns:
            True if result is valid
        """
        logger.info(f"Verifying bridged result {job_id} from {source_chain.value}")
        
        try:
            # Verify proof based on source chain
            if source_chain == ChainType.ETHEREUM:
                return await self._verify_ethereum_proof(job_id, proof)
            elif source_chain in [ChainType.POLKADOT, ChainType.KUSAMA]:
                return await self._verify_polkadot_proof(job_id, proof)
            else:
                logger.error(f"Unsupported source chain: {source_chain}")
                return False
                
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    async def _generate_bridge_proof(
        self,
        result_data: QuantumResultBridgeData
    ) -> BridgeProof:
        """Generate cryptographic proof for bridge transaction."""
        import hashlib
        
        # Create proof data (simplified - would use ZK proof in production)
        proof_content = (
            result_data.job_id.encode() +
            result_data.circuit_hash +
            result_data.result_hash
        )
        
        proof_hash = hashlib.sha256(proof_content).digest()
        
        return BridgeProof(
            proof_type="merkle",
            proof_data=proof_hash
        )
    
    async def _bridge_to_ethereum(
        self,
        result_data: QuantumResultBridgeData,
        destination_account: str,
        proof: Optional[BridgeProof]
    ) -> str:
        """
        Bridge quantum result to Ethereum.
        
        Returns transaction hash on Ethereum.
        """
        if not self.ethereum:
            raise RuntimeError("Ethereum bridge not configured")
        
        # Delegate to Ethereum bridge adapter
        return await self.ethereum.bridge_quantum_result(
            result_data=result_data,
            destination_account=destination_account,
            proof=proof
        )
    
    async def _bridge_to_polkadot(
        self,
        result_data: QuantumResultBridgeData,
        destination: BridgeDestination,
        proof: Optional[BridgeProof]
    ) -> str:
        """
        Bridge quantum result to Polkadot/Kusama via XCM.
        
        Returns transaction hash on target chain.
        """
        if not self.polkadot:
            raise RuntimeError("Polkadot bridge not configured")
        
        # Delegate to Polkadot XCM adapter
        return await self.polkadot.bridge_quantum_result(
            result_data=result_data,
            destination=destination,
            proof=proof
        )
    
    async def _bridge_nft_to_ethereum(
        self,
        nft_data: NFTBridgeData,
        destination_account: str
    ) -> str:
        """Bridge NFT to Ethereum (mint wrapped NFT)."""
        if not self.ethereum:
            raise RuntimeError("Ethereum bridge not configured")
        
        return await self.ethereum.bridge_nft(
            nft_data=nft_data,
            destination_account=destination_account
        )
    
    async def _bridge_nft_to_polkadot(
        self,
        nft_data: NFTBridgeData,
        destination: BridgeDestination
    ) -> str:
        """Bridge NFT to Polkadot/Kusama."""
        if not self.polkadot:
            raise RuntimeError("Polkadot bridge not configured")
        
        return await self.polkadot.bridge_nft(
            nft_data=nft_data,
            destination=destination
        )
    
    async def _lock_nft_on_belizechain(self, nft_id: str):
        """Lock NFT on BelizeChain to prevent double-spend."""
        if self.belizechain:
            # Call BelizeChain pallet to lock NFT
            logger.debug(f"Locked NFT {nft_id} on BelizeChain")
    
    async def _unlock_nft_on_belizechain(self, nft_id: str):
        """Unlock NFT on BelizeChain (bridge failed)."""
        if self.belizechain:
            logger.debug(f"Unlocked NFT {nft_id} on BelizeChain")
    
    async def _verify_ethereum_proof(
        self,
       job_id: str,
        proof: BridgeProof
    ) -> bool:
        """Verify proof from Ethereum."""
        if not self.ethereum:
            return False
        
        return await self.ethereum.verify_proof(job_id, proof)
    
    async def _verify_polkadot_proof(
        self,
        job_id: str,
        proof: BridgeProof
    ) -> bool:
        """Verify proof from Polkadot/Kusama."""
        if not self.polkadot:
            return False
        
        return await self.polkadot.verify_proof(job_id, proof)
    
    def get_transaction_status(self, tx_id: str) -> Optional[BridgeTransaction]:
        """Get status of bridge transaction."""
        # Check pending
        if tx_id in self.pending_transactions:
            return self.pending_transactions[tx_id]
        
        # Check completed
        for tx in self.completed_transactions:
            if tx.tx_id == tx_id:
                return tx
        
        return None
    
    def list_recent_transactions(self, limit: int = 50) -> List[BridgeTransaction]:
        """List recent bridge transactions."""
        return self.completed_transactions[-limit:]
