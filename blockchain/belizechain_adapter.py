"""
BelizeChain Blockchain Adapter for Kinich Quantum Computing

This module provides Python integration between Kinich quantum nodes and the
BelizeChain Substrate blockchain, enabling on-chain quantum job tracking,
result verification, and achievement NFT minting.

IMPORTANT: Uses u8 index pattern for Substrate v42 compatibility.
All enum parameters must be converted to numeric indices using quantum_indices module.
"""

import asyncio
import hashlib
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

try:
    from substrateinterface import SubstrateInterface, Keypair
    from substrateinterface.exceptions import SubstrateRequestException
    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False
    logging.warning("substrate-interface not installed. Blockchain integration disabled.")

from .quantum_indices import (
    QuantumBackendIndex,
    JobStatusIndex,
    VerificationStatusIndex,
    AchievementTypeIndex,
    VerificationVoteIndex,
    ChainDestinationIndex,
)

logger = logging.getLogger(__name__)


# Legacy enum classes for backwards compatibility
# NOTE: Deprecated - use quantum_indices module constants instead
class JobStatus(Enum):
    """[DEPRECATED] Use JobStatusIndex from quantum_indices module."""
    PENDING = JobStatusIndex.PENDING
    RUNNING = JobStatusIndex.RUNNING
    COMPLETED = JobStatusIndex.COMPLETED
    FAILED = JobStatusIndex.FAILED
    CANCELLED = JobStatusIndex.CANCELLED


class VerificationStatus(Enum):
    """[DEPRECATED] Use VerificationStatusIndex from quantum_indices module."""
    UNVERIFIED = VerificationStatusIndex.UNVERIFIED
    VERIFYING = VerificationStatusIndex.VERIFYING
    VERIFIED = VerificationStatusIndex.VERIFIED
    FAILED = VerificationStatusIndex.FAILED


class QuantumBackend(Enum):
    """[DEPRECATED] Use QuantumBackendIndex from quantum_indices module."""
    AZURE_IONQ = QuantumBackendIndex.AZURE_IONQ
    AZURE_QUANTINUUM = QuantumBackendIndex.AZURE_QUANTINUUM
    AZURE_RIGETTI = QuantumBackendIndex.AZURE_RIGETTI
    IBM_QUANTUM = QuantumBackendIndex.IBM_QUANTUM
    QISKIT = QuantumBackendIndex.QISKIT
    SPINQ_GEMINI = QuantumBackendIndex.SPINQ_GEMINI
    SPINQ_TRIANGULUM = QuantumBackendIndex.SPINQ_TRIANGULUM
    OTHER = QuantumBackendIndex.OTHER
    
    @classmethod
    def from_string(cls, backend_str: str) -> 'QuantumBackend':
        """Convert backend string to enum (DEPRECATED - returns index value)."""
        index = QuantumBackendIndex.from_string(backend_str)
        for member in cls:
            if member.value == index:
                return member
        return cls.OTHER


class AchievementType(Enum):
    """[DEPRECATED] Use AchievementTypeIndex from quantum_indices module."""
    FIRST_QUANTUM_JOB = AchievementTypeIndex.FIRST_QUANTUM_JOB
    GROVER_ALGORITHM = AchievementTypeIndex.GROVER_ALGORITHM
    SHOR_ALGORITHM = AchievementTypeIndex.SHOR_ALGORITHM
    QUANTUM_FOURIER_TRANSFORM = AchievementTypeIndex.QUANTUM_FOURIER_TRANSFORM
    VQE_ALGORITHM = AchievementTypeIndex.VQE_ALGORITHM
    QAOA_ALGORITHM = AchievementTypeIndex.QAOA_ALGORITHM
    ACCURACY_95 = AchievementTypeIndex.ACCURACY_95
    ACCURACY_99 = AchievementTypeIndex.ACCURACY_99
    VOLUME_CONTRIBUTOR_100 = AchievementTypeIndex.VOLUME_CONTRIBUTOR_100
    VOLUME_CONTRIBUTOR_1000 = AchievementTypeIndex.VOLUME_CONTRIBUTOR_1000
    ERROR_MITIGATION_CHAMPION = AchievementTypeIndex.ERROR_MITIGATION_CHAMPION
    CUSTOM = AchievementTypeIndex.CUSTOM


@dataclass
class QuantumJob:
    """On-chain quantum job record."""
    job_id: str
    submitter: str
    backend: QuantumBackend
    circuit_hash: bytes
    num_qubits: int
    circuit_depth: int
    num_shots: int
    status: JobStatus
    submission_time: int
    completion_time: Optional[int]
    result_hash: Optional[bytes]
    verification_status: VerificationStatus
    dalla_cost: int
    executor: Optional[str]


@dataclass
class QuantumResult:
    """On-chain quantum result record."""
    job_id: str
    result_data_hash: bytes
    verification_proof: bytes
    accuracy_score: int
    validator: str
    recorded_at: int


@dataclass
class QuantumAchievement:
    """Quantum achievement NFT."""
    nft_id: int
    job_id: str
    achievement_type: AchievementType
    owner: str
    metadata_uri: str
    minted_at: int
    transferable: bool


class BelizeChainAdapter:
    """
    Adapter for submitting quantum jobs to BelizeChain.
    
    Provides methods to:
    - Submit quantum jobs to blockchain
    - Record quantum results on-chain
    - Mint achievement NFTs
    - Query job status and statistics
    """
    
    def __init__(
        self,
        node_url: str = "ws://localhost:9944",
        keypair: Optional[Keypair] = None,
        keypair_seed: Optional[str] = None
    ):
        """
        Initialize BelizeChain adapter.
        
        Args:
            node_url: WebSocket URL of BelizeChain node
            keypair: Pre-initialized Substrate keypair
            keypair_seed: Seed phrase for keypair generation
        """
        if not SUBSTRATE_AVAILABLE:
            raise RuntimeError(
                "substrate-interface not installed. "
                "Install with: pip install substrate-interface"
            )
        
        self.node_url = node_url
        self.substrate: Optional[SubstrateInterface] = None
        
        # Initialize keypair
        if keypair:
            self.keypair = keypair
        elif keypair_seed:
            self.keypair = Keypair.create_from_mnemonic(keypair_seed)
        else:
            # Development keypair (Alice)
            self.keypair = Keypair.create_from_uri('//Alice')
        
        logger.info(f"Initialized BelizeChain adapter for {self.keypair.ss58_address}")
    
    async def connect(self) -> bool:
        """
        Connect to BelizeChain node.
        
        Returns:
            True if connected successfully
        """
        try:
            self.substrate = SubstrateInterface(url=self.node_url)
            logger.info(f"Connected to BelizeChain at {self.node_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to BelizeChain: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from BelizeChain node."""
        if self.substrate:
            self.substrate.close()
            self.substrate = None
            logger.info("Disconnected from BelizeChain")
    
    def _ensure_connected(self):
        """Ensure connection to blockchain."""
        if not self.substrate:
            raise RuntimeError("Not connected to BelizeChain. Call connect() first.")
    
    async def submit_quantum_job(
        self,
        job_id: str,
        backend: str,
        circuit_hash: bytes,
        num_qubits: int,
        circuit_depth: int,
        num_shots: int = 1024,
        backend_index: Optional[int] = None
    ) -> Optional[str]:
        """
        Submit quantum job to blockchain.
        
        Args:
            job_id: Unique job identifier (UUID)
            backend: Quantum backend name (e.g., "azure_ionq") [DEPRECATED]
            circuit_hash: SHA-256 hash of circuit
            num_qubits: Number of qubits
            circuit_depth: Circuit depth
            num_shots: Number of measurements
            backend_index: Backend index (0-7). If provided, overrides 'backend' parameter.
        
        Returns:
            Transaction hash if successful, None otherwise
        
        Note:
            Prefer using backend_index directly for Substrate v42 compatibility.
            The 'backend' string parameter is deprecated and will be removed in v2.0.
        """
        self._ensure_connected()
        
        try:
            # Convert backend to index (prefer backend_index if provided)
            if backend_index is not None:
                if not QuantumBackendIndex.validate(backend_index):
                    raise ValueError(f"Invalid backend index {backend_index}. Must be 0-7.")
                final_backend_index = backend_index
            else:
                # Legacy: convert backend string to index
                final_backend_index = QuantumBackendIndex.from_string(backend)
                logger.warning(
                    f"Using deprecated 'backend' string parameter. "
                    f"Use backend_index={final_backend_index} instead."
                )
            
            # Create extrinsic call
            call = self.substrate.compose_call(
                call_module='Quantum',
                call_function='submit_quantum_job',
                call_params={
                    'job_id': job_id.encode('utf-8'),
                    'backend_index': final_backend_index,  # Changed from 'backend'
                    'circuit_hash': circuit_hash,
                    'num_qubits': num_qubits,
                    'circuit_depth': circuit_depth,
                    'num_shots': num_shots,
                }
            )
            
            # Sign and submit
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=self.keypair
            )
            
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True
            )
            
            if receipt.is_success:
                logger.info(f"✅ Submitted quantum job {job_id} to blockchain")
                logger.info(f"   Transaction: {receipt.extrinsic_hash}")
                return receipt.extrinsic_hash
            else:
                logger.error(f"❌ Failed to submit job {job_id}: {receipt.error_message}")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting quantum job: {e}")
            return None
    
    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus = None,
        status_index: Optional[int] = None
    ) -> Optional[str]:
        """
        Update quantum job status on blockchain.
        
        Args:
            job_id: Job identifier
            status: JobStatus enum [DEPRECATED]
            status_index: Status index (0-4). Preferred over status enum.
        
        Returns:
            Transaction hash if successful
        
        Note:
            Use status_index directly for Substrate v42 compatibility.
        """
        self._ensure_connected()
        
        try:
            # Convert status to index (prefer status_index if provided)
            if status_index is not None:
                if not JobStatusIndex.validate(status_index):
                    raise ValueError(f"Invalid status index {status_index}. Must be 0-4.")
                final_status_index = status_index
            elif status is not None:
                final_status_index = status.value
            else:
                raise ValueError("Either status or status_index must be provided")
            
            call = self.substrate.compose_call(
                call_module='Quantum',
                call_function='update_job_status',
                call_params={
                    'job_id': job_id.encode('utf-8'),
                    'new_status_index': final_status_index,  # Changed from 'new_status'
                }
            )
            
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=self.keypair
            )
            
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True
            )
            
            if receipt.is_success:
                logger.info(f"✅ Updated job {job_id} status to {status.name}")
                return receipt.extrinsic_hash
            else:
                logger.error(f"❌ Failed to update job status: {receipt.error_message}")
                return None
                
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
            return None
    
    async def record_quantum_result(
        self,
        job_id: str,
        result_data: Dict[str, Any],
        accuracy_score: int
    ) -> Optional[str]:
        """
        Record quantum result on blockchain.
        
        Args:
            job_id: Job identifier
            result_data: Complete result data (will be hashed)
            accuracy_score: Accuracy score (0-100)
        
        Returns:
            Transaction hash if successful
        """
        self._ensure_connected()
        
        try:
            # Hash result data
            result_json = json.dumps(result_data, sort_keys=True)
            result_hash = hashlib.sha256(result_json.encode()).digest()
            
            # Generate simplified verification proof (placeholder)
            # In production, this would be a real ZK-SNARK proof
            verification_proof = self._generate_verification_proof(result_data)
            
            call = self.substrate.compose_call(
                call_module='Quantum',
                call_function='record_quantum_result',
                call_params={
                    'job_id': job_id.encode('utf-8'),
                    'result_data_hash': result_hash,
                    'verification_proof': verification_proof,
                    'accuracy_score': accuracy_score,
                }
            )
            
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=self.keypair
            )
            
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True
            )
            
            if receipt.is_success:
                logger.info(f"✅ Recorded result for job {job_id} (accuracy: {accuracy_score}%)")
                return receipt.extrinsic_hash
            else:
                logger.error(f"❌ Failed to record result: {receipt.error_message}")
                return None
                
        except Exception as e:
            logger.error(f"Error recording quantum result: {e}")
            return None
    
    def _generate_verification_proof(self, result_data: Dict[str, Any]) -> bytes:
        """
        Generate simplified verification proof.
        
        In production, this would generate a real ZK-SNARK proof.
        Currently generates a cryptographic signature as placeholder.
        
        Args:
            result_data: Result data to prove
        
        Returns:
            Verification proof bytes
        """
        result_json = json.dumps(result_data, sort_keys=True)
        proof_data = f"{result_json}:{self.keypair.ss58_address}".encode()
        
        # Simple HMAC-style proof (placeholder for real ZK proof)
        proof_hash = hashlib.sha256(proof_data).digest()
        
        return proof_hash
    
    async def mint_achievement_nft(
        self,
        job_id: str,
        achievement_type: AchievementType = None,
        metadata: Dict[str, Any] = None,
        transferable: bool = True,
        achievement_index: Optional[int] = None
    ) -> Optional[int]:
        """
        Mint quantum achievement NFT.
        
        Args:
            job_id: Associated quantum job
            achievement_type: AchievementType enum [DEPRECATED]
            metadata: NFT metadata (will be uploaded to IPFS)
            transferable: Whether NFT can be transferred
            achievement_index: Achievement type index (0-11). Preferred over enum.
        
        Returns:
            NFT ID if successful, None otherwise
        
        Note:
            Use achievement_index directly for Substrate v42 compatibility.
        """
        self._ensure_connected()
        
        try:
            # Convert achievement type to index (prefer achievement_index if provided)
            if achievement_index is not None:
                if not AchievementTypeIndex.validate(achievement_index):
                    raise ValueError(f"Invalid achievement index {achievement_index}. Must be 0-11.")
                final_achievement_index = achievement_index
            elif achievement_type is not None:
                final_achievement_index = achievement_type.value
            else:
                raise ValueError("Either achievement_type or achievement_index must be provided")
            
            # Upload metadata to IPFS (placeholder - would use real IPFS in production)
            if metadata is None:
                metadata = {}
            metadata_uri = await self._upload_to_ipfs(metadata)
            
            call = self.substrate.compose_call(
                call_module='Quantum',
                call_function='mint_achievement_nft',
                call_params={
                    'job_id': job_id.encode('utf-8'),
                    'achievement_type_index': final_achievement_index,  # Changed from 'achievement_type'
                    'metadata_uri': metadata_uri.encode('utf-8'),
                    'transferable': transferable,
                }
            )
            
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=self.keypair
            )
            
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True
            )
            
            if receipt.is_success:
                # Extract NFT ID from events
                for event in receipt.triggered_events:
                    if event.value['module_id'] == 'Quantum' and \
                       event.value['event_id'] == 'AchievementNFTMinted':
                        nft_id = event.value['attributes']['nft_id']
                        logger.info(f"🏆 Minted achievement NFT #{nft_id} for job {job_id}")
                        return nft_id
                
                logger.warning("NFT minted but ID not found in events")
                return None
            else:
                logger.error(f"❌ Failed to mint NFT: {receipt.error_message}")
                return None
                
        except Exception as e:
            logger.error(f"Error minting achievement NFT: {e}")
            return None
    
    async def _upload_to_ipfs(self, metadata: Dict[str, Any]) -> str:
        """
        Upload metadata to IPFS.
        
        Placeholder implementation. In production, would use ipfshttpclient
        or Arweave for permanent storage.
        
        Args:
            metadata: Metadata dictionary
        
        Returns:
            IPFS URI (ipfs://...)
        """
        # Placeholder: return deterministic hash
        metadata_json = json.dumps(metadata, sort_keys=True)
        content_hash = hashlib.sha256(metadata_json.encode()).hexdigest()[:46]
        
        # In production: upload to actual IPFS and return real CID
        # ipfs_client.add_json(metadata)
        
        return f"ipfs://Qm{content_hash}"
    
    async def get_job_status(self, job_id: str) -> Optional[QuantumJob]:
        """
        Fetch quantum job from blockchain.
        
        Args:
            job_id: Job identifier
        
        Returns:
            QuantumJob if found, None otherwise
        """
        self._ensure_connected()
        
        try:
            result = self.substrate.query(
                module='Quantum',
                storage_function='QuantumJobs',
                params=[job_id.encode('utf-8')]
            )
            
            if result.value:
                job_data = result.value
                return QuantumJob(
                    job_id=job_id,
                    submitter=job_data['submitter'],
                    backend=QuantumBackend(job_data['backend']),
                    circuit_hash=bytes(job_data['circuit_hash']),
                    num_qubits=job_data['num_qubits'],
                    circuit_depth=job_data['circuit_depth'],
                    num_shots=job_data['num_shots'],
                    status=JobStatus(job_data['status']),
                    submission_time=job_data['submission_time'],
                    completion_time=job_data.get('completion_time'),
                    result_hash=bytes(job_data['result_hash']) if job_data.get('result_hash') else None,
                    verification_status=VerificationStatus(job_data['verification_status']),
                    dalla_cost=job_data['dalla_cost'],
                    executor=job_data.get('executor'),
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching job status: {e}")
            return None
    
    async def get_account_stats(self, account: str) -> Optional[Dict[str, Any]]:
        """
        Get quantum statistics for account.
        
        Args:
            account: Account address (SS58 format)
        
        Returns:
            Statistics dictionary if found
        """
        self._ensure_connected()
        
        try:
            result = self.substrate.query(
                module='Quantum',
                storage_function='AccountStats',
                params=[account]
            )
            
            if result.value:
                return {
                    'total_jobs': result.value['total_jobs'],
                    'completed_jobs': result.value['completed_jobs'],
                    'failed_jobs': result.value['failed_jobs'],
                    'total_spent': result.value['total_spent'],
                    'total_qubits': result.value['total_qubits'],
                    'total_shots': result.value['total_shots'],
                    'nfts_earned': result.value['nfts_earned'],
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching account stats: {e}")
            return None
    
    async def watch_job_events(
        self,
        job_id: Optional[str] = None,
        callback=None
    ):
        """
        Watch blockchain events for quantum jobs.
        
        Args:
            job_id: Specific job to watch (None for all jobs)
            callback: Async callback function for events
        """
        self._ensure_connected()
        
        logger.info(f"👀 Watching quantum job events{f' for {job_id}' if job_id else ''}...")
        
        def event_handler(obj, update_nr, subscription_id):
            """Handle incoming blockchain events."""
            block_number = obj['header']['number']
            
            for event in obj['events']:
                if event['module_id'] == 'Quantum':
                    event_data = {
                        'block': block_number,
                        'event': event['event_id'],
                        'attributes': event['attributes'],
                    }
                    
                    # Filter by job_id if specified
                    if job_id:
                        event_job_id = event['attributes'].get('job_id', b'').decode('utf-8')
                        if event_job_id != job_id:
                            continue
                    
                    logger.info(f"📡 Quantum event: {event['event_id']} at block {block_number}")
                    
                    if callback:
                        asyncio.create_task(callback(event_data))
        
        # Subscribe to new block headers
        self.substrate.subscribe_block_headers(event_handler)


# Convenience function for testing
async def test_blockchain_connection():
    """Test blockchain connection and basic operations."""
    adapter = BelizeChainAdapter()
    
    if await adapter.connect():
        print("✅ Connected to BelizeChain")
        print(f"   Account: {adapter.keypair.ss58_address}")
        
        # Test queries
        stats = await adapter.get_account_stats(adapter.keypair.ss58_address)
        if stats:
            print(f"   Total jobs: {stats['total_jobs']}")
            print(f"   NFTs earned: {stats['nfts_earned']}")
        
        await adapter.disconnect()
        return True
    else:
        print("❌ Failed to connect to BelizeChain")
        return False


if __name__ == "__main__":
    asyncio.run(test_blockchain_connection())
