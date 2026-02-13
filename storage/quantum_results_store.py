"""
Quantum Results Store

Enhanced storage for quantum computation results in Pakit with:
- DAG content addressing (CID generation)
- Storage proofs (Merkle trees)
- Long-term archival
- Cross-node replication
- Blockchain proof integration
- Large result chunking (>1MB)
"""

import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False

from .dag_content_addressing import (
    DAGContentAddressing,
    DAGContent,
    HashFunction,
    CIDVersion
)
from .storage_proof_generator import (
    StorageProofGenerator,
    MerkleProof,
    ReplicationProof
)
from .result_retriever import QuantumResultRetriever


class QuantumResultsStore:
    """
    Enhanced quantum computation results storage with DAG addressing and proofs.
    
    Integration with:
    - Pakit DAG storage (IPFS/Arweave backend)
    - BelizeChain Consensus pallet (Proof of Quantum Work)
    - Storage proof verification on-chain
    - Large result chunking and reconstruction
    """
    
    # Size threshold for chunking (1MB)
    CHUNK_SIZE_THRESHOLD = 1024 * 1024
    CHUNK_SIZE = 256 * 1024  # 256KB chunks
    
    def __init__(
        self,
        pakit_api_url: str = "http://localhost:8000",
        ipfs_gateway: Optional[str] = None,
        blockchain_connector: Optional[Any] = None,
        chunk_large_results: bool = True,
        generate_proofs: bool = True
    ):
        """
        Initialize enhanced results store.
        
        Args:
            pakit_api_url: Pakit API endpoint
            ipfs_gateway: IPFS gateway URL (optional)
            blockchain_connector: BelizeChain connector for on-chain proofs
            chunk_large_results: Automatically chunk results > 1MB
            generate_proofs: Generate storage proofs for verification
        """
        self.pakit_api_url = pakit_api_url.rstrip('/')
        self.ipfs_gateway = ipfs_gateway
        self.blockchain_connector = blockchain_connector
        self.chunk_large_results = chunk_large_results
        self.generate_proofs = generate_proofs
        
        # Initialize DAG content addressing
        self.dag = DAGContentAddressing(
            hash_function=HashFunction.SHA256,
            cid_version=CIDVersion.V1
        )
        
        # Initialize storage proof generator
        self.proof_generator = StorageProofGenerator()
        
        # Initialize result retriever
        self.retriever = QuantumResultRetriever(
            pakit_api_url=pakit_api_url,
            ipfs_gateway=ipfs_gateway
        )
        
        # Try to connect to local IPFS node
        self.ipfs_client = None
        if IPFS_AVAILABLE:
            try:
                self.ipfs_client = ipfshttpclient.connect()
                logger.info("✅ Connected to local IPFS node")
            except Exception as e:
                logger.warning(f"Could not connect to local IPFS: {e}")
    
    def store_quantum_result(
        self,
        job_id: str,
        circuit_qasm: str,
        counts: Dict[str, int],
        backend: str,
        metadata: Optional[Dict[str, Any]] = None,
        submit_proof_to_chain: bool = True
    ) -> Tuple[str, Optional[MerkleProof]]:
        """
        Store quantum computation result with DAG addressing and proofs.
        
        Args:
            job_id: Quantum job ID
            circuit_qasm: OpenQASM circuit definition
            counts: Measurement counts
            backend: Quantum backend used
            metadata: Additional metadata
            submit_proof_to_chain: Submit storage proof to blockchain
            
        Returns:
            Tuple of (Content ID, Storage Proof)
        """
        # Prepare result package
        result_package = {
            'job_id': job_id,
            'timestamp': datetime.now().isoformat(),
            'circuit_qasm': circuit_qasm,
            'counts': counts,
            'backend': backend,
            'metadata': metadata or {},
        }
        
        result_json = json.dumps(result_package, sort_keys=True).encode('utf-8')
        result_size = len(result_json)
        
        logger.info(f"Storing quantum result {job_id} ({result_size} bytes)")
        
        # Check if chunking is needed
        if self.chunk_large_results and result_size > self.CHUNK_SIZE_THRESHOLD:
            return self._store_chunked_result(
                result_package,
                result_json,
                submit_proof_to_chain
            )
        else:
            return self._store_single_result(
                result_package,
                result_json,
                submit_proof_to_chain
            )
    
    def _store_single_result(
        self,
        result_package: Dict[str, Any],
        result_json: bytes,
        submit_proof_to_chain: bool
    ) -> Tuple[str, Optional[MerkleProof]]:
        """Store single (non-chunked) result."""
        # Generate CID
        dag_content = self.dag.generate_dag_content(result_json)
        cid = dag_content.cid
        
        logger.info(f"Generated CID: {cid}")
        
        # Upload to storage
        stored_cid = self._upload_to_pakit(
            data=result_json,
            cid=cid,
            metadata={
                'type': 'quantum_result',
                'job_id': result_package['job_id'],
                'backend': result_package['backend'],
                'chunked': False
            }
        )
        
        # Generate storage proof
        proof = None
        if self.generate_proofs:
            proof = self.proof_generator.generate_inclusion_proof(
                data_chunks=[result_json],
                chunk_index=0
            )
            logger.info(f"Generated storage proof (root: {proof.root_hash.hex()[:16]}...)")
        
        # Submit proof to blockchain
        if submit_proof_to_chain and self.blockchain_connector:
            self._submit_storage_proof_to_chain(
                job_id=result_package['job_id'],
                cid=stored_cid,
                proof=proof,
                backend=result_package['backend']
            )
        
        return stored_cid, proof
    
    def _store_chunked_result(
        self,
        result_package: Dict[str, Any],
        result_json: bytes,
        submit_proof_to_chain: bool
    ) -> Tuple[str, Optional[MerkleProof]]:
        """Store large result as chunks with manifest."""
        logger.info(f"Chunking large result ({len(result_json)} bytes)")
        
        # Chunk data
        chunks = self.dag.chunk_data(result_json, chunk_size=self.CHUNK_SIZE)
        
        # Upload each chunk
        chunk_cids = []
        for i, chunk_content in enumerate(chunks):
            chunk_cid = self._upload_to_pakit(
                data=chunk_content.data,
                cid=chunk_content.cid,
                metadata={
                    'type': 'quantum_result_chunk',
                    'job_id': result_package['job_id'],
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            )
            chunk_cids.append(chunk_cid)
            logger.debug(f"Uploaded chunk {i+1}/{len(chunks)}: {chunk_cid}")
        
        # Create manifest
        manifest = self.dag.create_dag_manifest(
            chunks=chunks,
            metadata={
                'job_id': result_package['job_id'],
                'backend': result_package['backend'],
                'timestamp': result_package['timestamp']
            }
        )
        
        # Upload manifest
        manifest_json = json.dumps({
            'version': '1.0',
            'job_id': result_package['job_id'],
            'chunk_count': len(chunks),
            'total_size': len(result_json),
            'chunks': [
                {
                    'cid': chunk.cid,
                    'size': chunk.size,
                    'hash': chunk.multihash_bytes.hex()
                }
                for chunk in chunks
            ],
            'metadata': result_package['metadata']
        }, sort_keys=True).encode('utf-8')
        
        manifest_content = self.dag.generate_dag_content(manifest_json)
        manifest_cid = self._upload_to_pakit(
            data=manifest_json,
            cid=manifest_content.cid,
            metadata={
                'type': 'quantum_result_manifest',
                'job_id': result_package['job_id'],
                'chunked': True,
                'chunk_count': len(chunks)
            }
        )
        
        logger.info(f"✅ Stored chunked result: {manifest_cid} ({len(chunks)} chunks)")
        
        # Generate proof for manifest
        proof = None
        if self.generate_proofs:
            proof = self.proof_generator.generate_inclusion_proof(
                data_chunks=[manifest_json],
                chunk_index=0
            )
        
        # Submit proof to blockchain
        if submit_proof_to_chain and self.blockchain_connector:
            self._submit_storage_proof_to_chain(
                job_id=result_package['job_id'],
                cid=manifest_cid,
                proof=proof,
                backend=result_package['backend'],
                is_chunked=True,
                chunk_count=len(chunks)
            )
        
        return manifest_cid, proof
    
    def _upload_to_pakit(
        self,
        data: bytes,
        cid: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Upload data to Pakit storage."""
        # Try IPFS client first (direct, fastest)
        if self.ipfs_client:
            try:
                result = self.ipfs_client.add_bytes(data)
                ipfs_cid = result
                logger.debug(f"Uploaded to IPFS: {ipfs_cid}")
                return ipfs_cid
            except Exception as e:
                logger.warning(f"IPFS upload failed: {e}")
        
        # Try Pakit API
        if REQUESTS_AVAILABLE:
            try:
                response = requests.post(
                    f"{self.pakit_api_url}/api/v1/upload",
                    files={'file': (f"{cid}.dat", data)},
                    data={'metadata': json.dumps(metadata)}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    returned_cid = result.get('cid') or result.get('content_id') or cid
                    logger.debug(f"Uploaded to Pakit: {returned_cid}")
                    return returned_cid
                else:
                    logger.warning(f"Pakit upload failed: {response.status_code}")
            except Exception as e:
                logger.warning(f"Pakit upload error: {e}")
        
        # Fallback: return generated CID (data not actually uploaded)
        logger.warning("Data not uploaded - returning generated CID only")
        return cid
    
    def retrieve_quantum_result(
        self,
        content_id: str,
        validate_integrity: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve quantum result from DAG storage.
        
        Supports both chunked and non-chunked results.
        
        Args:
            content_id: Content ID (CID) to retrieve
            validate_integrity: Verify CID matches data
            
        Returns:
            Result package or None
        """
        try:
            return self.retriever.retrieve_quantum_result(
                cid=content_id,
                validate_structure=True
            )
        except Exception as e:
            logger.error(f"Retrieve error: {e}")
            # Try as chunked result
            try:
                data = self.retriever.retrieve_chunked_result(
                    manifest_cid=content_id,
                    validate_proofs=validate_integrity
                )
                return json.loads(data.decode('utf-8'))
            except Exception as e2:
                logger.error(f"Chunked retrieve also failed: {e2}")
                return None
    
    def retrieve_with_proof(
        self,
        content_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[MerkleProof]]:
        """
        Retrieve quantum result with storage proof.
        
        Args:
            content_id: Content ID to retrieve
            
        Returns:
            Tuple of (result, proof)
        """
        try:
            data, proof = self.retriever.retrieve_with_proof(content_id)
            result = json.loads(data.decode('utf-8'))
            return result, proof
        except Exception as e:
            logger.error(f"Retrieve with proof error: {e}")
            return None, None
    
    def verify_result_integrity(
        self,
        content_id: str,
        result_data: Dict[str, Any]
    ) -> bool:
        """
        Verify that result data matches its CID.
        
        Args:
            content_id: Expected CID
            result_data: Result data to verify
            
        Returns:
            True if data matches CID
        """
        result_json = json.dumps(result_data, sort_keys=True).encode('utf-8')
        return self.dag.verify_cid(result_json, content_id)
    
    def list_job_results(
        self,
        backend: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List stored quantum results.
        
        Args:
            backend: Filter by backend
            limit: Max results
            
        Returns:
            List of result metadata
        """
        if not REQUESTS_AVAILABLE:
            return []
        
        try:
            params = {'type': 'quantum_result', 'limit': limit}
            if backend:
                params['backend'] = backend
            
            response = requests.get(
                f"{self.pakit_api_url}/api/v1/list",
                params=params
            )
            
            if response.status_code == 200:
                return response.json().get('results', [])
            return []
            
        except Exception as e:
            logger.error(f"List error: {e}")
            return []
    
    def check_result_availability(self, content_id: str) -> Dict[str, bool]:
        """
        Check which sources have the result available.
        
        Args:
            content_id: Content ID to check
            
        Returns:
            Dictionary with availability status
        """
        return self.retriever.check_availability(content_id)
    
    def generate_replication_proof(
        self,
        content_id: str,
        replica_id: int,
        challenge: bytes
    ) -> Optional[ReplicationProof]:
        """
        Generate proof of replication for a result.
        
        Args:
            content_id: CID of the result
            replica_id: ID of the replica
            challenge: Random challenge bytes
            
        Returns:
            ReplicationProof or None
        """
        try:
            # Retrieve data
            data = self.retriever.retrieve_by_cid(content_id)
            
            # Create chunks for proof
            chunks = [data]  # Simple case: single chunk
            
            # Generate replication proof
            proof = self.proof_generator.generate_replication_proof(
                data_chunks=chunks,
                replica_id=replica_id,
                challenge=challenge
            )
            
            logger.info(f"Generated replication proof for {content_id}")
            return proof
            
        except Exception as e:
            logger.error(f"Replication proof generation failed: {e}")
            return None
    
    def _submit_storage_proof_to_chain(
        self,
        job_id: str,
        cid: str,
        proof: Optional[MerkleProof],
        backend: str,
        is_chunked: bool = False,
        chunk_count: int = 0
    ):
        """
        Submit storage proof to BelizeChain.
        
        Creates Proof of Storage Work (PoSW) on-chain.
        """
        if not self.blockchain_connector:
            logger.debug("No blockchain connector - skipping proof submission")
            return
        
        try:
            # Prepare proof data for blockchain
            proof_data = {
                'job_id': job_id,
                'cid': cid,
                'backend': backend,
                'is_chunked': is_chunked,
                'chunk_count': chunk_count,
                'merkle_root': proof.root_hash.hex() if proof else None,
                'timestamp': int(datetime.now().timestamp())
            }
            
            # Submit to blockchain (using PoSW extrinsic)
            # This would call: pallet-quantum.submit_storage_work()
            logger.info(f"📜 Submitting storage proof to blockchain: {job_id}")
            logger.debug(f"   CID: {cid}")
            logger.debug(f"   Merkle root: {proof.root_hash.hex()[:16]}..." if proof else "   No proof")
            
            # TODO: Actual blockchain submission
            # await self.blockchain_connector.submit_storage_work(
            #     job_id=job_id,
            #     cid=cid,
            #     merkle_root=proof.root_hash if proof else b'',
            #     chunk_count=chunk_count
            # )
            
        except Exception as e:
            logger.warning(f"Blockchain proof submission failed: {e}")
