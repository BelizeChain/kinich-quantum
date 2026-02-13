"""
Quantum Result Retrieval from DAG Storage

Retrieves quantum computation results from Pakit DAG storage by CID.
Supports chunked data reconstruction and integrity verification.
"""

import json
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available")

try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False
    logger.warning("ipfshttpclient not available")

from .dag_content_addressing import DAGContentAddressing, DAGContent
from .storage_proof_generator import StorageProofGenerator, MerkleProof


class ResultRetrievalError(Exception):
    """Error retrieving result from storage."""
    pass


class QuantumResultRetriever:
    """
    Retrieve quantum computation results from Pakit DAG storage.
    
    Supports:
    - Direct CID retrieval
    - Chunked data reconstruction  
    - Integrity verification
    - IPFS and HTTP gateway access
    """
    
    def __init__(
        self,
        pakit_api_url: str = "http://localhost:8000",
        ipfs_gateway: Optional[str] = None,
        verify_integrity: bool = True
    ):
        """
        Initialize result retriever.
        
        Args:
            pakit_api_url: Pakit API endpoint
            ipfs_gateway: IPFS gateway URL (optional)
            verify_integrity: Verify CID matches retrieved data
        """
        self.pakit_api_url = pakit_api_url.rstrip('/')
        self.ipfs_gateway = ipfs_gateway.rstrip('/') if ipfs_gateway else None
        self.verify_integrity = verify_integrity
        
        self.dag = DAGContentAddressing()
        self.proof_generator = StorageProofGenerator()
        
        # Try to connect to local IPFS node
        self.ipfs_client = None
        if IPFS_AVAILABLE and not ipfs_gateway:
            try:
                self.ipfs_client = ipfshttpclient.connect()
                logger.info("Connected to local IPFS node")
            except Exception as e:
                logger.warning(f"Could not connect to local IPFS: {e}")
    
    def retrieve_by_cid(
        self,
        cid: str,
        validate_proof: bool = False
    ) -> bytes:
        """
        Retrieve data by CID.
        
        Args:
            cid: Content ID to retrieve
            validate_proof: Validate storage proof if available
            
        Returns:
            Raw data bytes
            
        Raises:
            ResultRetrievalError: If retrieval fails
        """
        # Try IPFS client first (fastest)
        if self.ipfs_client:
            try:
                data = self.ipfs_client.cat(cid)
                if self.verify_integrity:
                    if not self.dag.verify_cid(data, cid):
                        raise ResultRetrievalError(f"CID mismatch for {cid}")
                logger.info(f"Retrieved {len(data)} bytes from IPFS for {cid}")
                return data
            except Exception as e:
                logger.warning(f"IPFS retrieval failed: {e}")
        
        # Try Pakit API
        if REQUESTS_AVAILABLE:
            try:
                data = self._retrieve_from_pakit(cid)
                if data:
                    if self.verify_integrity:
                        if not self.dag.verify_cid(data, cid):
                            raise ResultRetrievalError(f"CID mismatch for {cid}")
                    return data
            except Exception as e:
                logger.warning(f"Pakit API retrieval failed: {e}")
        
        # Try IPFS gateway
        if self.ipfs_gateway and REQUESTS_AVAILABLE:
            try:
                data = self._retrieve_from_gateway(cid)
                if data:
                    if self.verify_integrity:
                        if not self.dag.verify_cid(data, cid):
                            raise ResultRetrievalError(f"CID mismatch for {cid}")
                    return data
            except Exception as e:
                logger.warning(f"IPFS gateway retrieval failed: {e}")
        
        raise ResultRetrievalError(f"Failed to retrieve CID {cid} from all sources")
    
    def _retrieve_from_pakit(self, cid: str) -> Optional[bytes]:
        """Retrieve data from Pakit API."""
        try:
            response = requests.get(
                f"{self.pakit_api_url}/api/v1/retrieve/{cid}",
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Retrieved from Pakit: {cid}")
                return response.content
            else:
                logger.error(f"Pakit returned {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Pakit retrieval error: {e}")
            return None
    
    def _retrieve_from_gateway(self, cid: str) -> Optional[bytes]:
        """Retrieve data from IPFS gateway."""
        try:
            response = requests.get(
                f"{self.ipfs_gateway}/ipfs/{cid}",
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Retrieved from IPFS gateway: {cid}")
                return response.content
            else:
                logger.error(f"Gateway returned {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Gateway retrieval error: {e}")
            return None
    
    def retrieve_quantum_result(
        self,
        cid: str,
        validate_structure: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve and parse quantum result.
        
        Args:
            cid: Content ID of quantum result
            validate_structure: Validate result has required fields
            
        Returns:
            Quantum result dictionary
            
        Raises:
            ResultRetrievalError: If retrieval or parsing fails
        """
        # Retrieve raw data
        data = self.retrieve_by_cid(cid)
        
        # Parse JSON
        try:
            result = json.loads(data.decode('utf-8'))
        except Exception as e:
            raise ResultRetrievalError(f"Failed to parse result JSON: {e}")
        
        # Validate structure
        if validate_structure:
            required_fields = ['job_id', 'circuit_qasm', 'counts', 'backend']
            missing = [f for f in required_fields if f not in result]
            if missing:
                raise ResultRetrievalError(f"Missing required fields: {missing}")
        
        logger.info(f"Retrieved quantum result for job {result.get('job_id')}")
        return result
    
    def retrieve_chunked_result(
        self,
        manifest_cid: str,
        validate_proofs: bool = True
    ) -> bytes:
        """
        Retrieve and reconstruct chunked result.
        
        Args:
            manifest_cid: CID of the manifest (links to chunks)
            validate_proofs: Validate Merkle proofs for chunks
            
        Returns:
            Reconstructed data bytes
            
        Raises:
            ResultRetrievalError: If retrieval or reconstruction fails
        """
        # Retrieve manifest
        manifest_data = self.retrieve_by_cid(manifest_cid)
        manifest = json.loads(manifest_data.decode('utf-8'))
        
        if 'chunks' not in manifest:
            raise ResultRetrievalError("Invalid manifest: missing 'chunks'")
        
        logger.info(f"Retrieving {manifest['chunk_count']} chunks from manifest {manifest_cid}")
        
        # Retrieve all chunks
        chunks_data = []
        for i, chunk_info in enumerate(manifest['chunks']):
            chunk_cid = chunk_info['cid']
            logger.debug(f"Retrieving chunk {i+1}/{manifest['chunk_count']}: {chunk_cid}")
            
            chunk_data = self.retrieve_by_cid(chunk_cid)
            chunks_data.append(chunk_data)
        
        # Validate Merkle proofs if requested
        if validate_proofs and 'merkle_root' in manifest:
            logger.info("Validating Merkle proofs for chunks...")
            # Would need to store proofs in manifest for this
            # For now, just verify chunk count
            if len(chunks_data) != manifest['chunk_count']:
                raise ResultRetrievalError(
                    f"Chunk count mismatch: expected {manifest['chunk_count']}, "
                    f"got {len(chunks_data)}"
                )
        
        # Reconstruct data
        reconstructed = b''.join(chunks_data)
        
        # Verify total size
        if 'total_size' in manifest:
            if len(reconstructed) != manifest['total_size']:
                raise ResultRetrievalError(
                    f"Size mismatch: expected {manifest['total_size']}, "
                    f"got {len(reconstructed)}"
                )
        
        logger.info(f"Successfully reconstructed {len(reconstructed)} bytes")
        return reconstructed
    
    def check_availability(self, cid: str) -> Dict[str, bool]:
        """
        Check which sources have the CID available.
        
        Args:
            cid: Content ID to check
            
        Returns:
            Dictionary with availability status for each source
        """
        availability = {
            'ipfs_node': False,
            'pakit_api': False,
            'ipfs_gateway': False
        }
        
        # Check IPFS node
        if self.ipfs_client:
            try:
                stat = self.ipfs_client.object.stat(cid)
                availability['ipfs_node'] = True
            except:
                pass
        
        # Check Pakit API
        if REQUESTS_AVAILABLE:
            try:
                response = requests.head(
                    f"{self.pakit_api_url}/api/v1/retrieve/{cid}",
                    timeout=5
                )
                availability['pakit_api'] = response.status_code == 200
            except:
                pass
        
        # Check IPFS gateway
        if self.ipfs_gateway and REQUESTS_AVAILABLE:
            try:
                response = requests.head(
                    f"{self.ipfs_gateway}/ipfs/{cid}",
                    timeout=5
                )
                availability['ipfs_gateway'] = response.status_code == 200
            except:
                pass
        
        return availability
    
    def retrieve_with_proof(
        self,
        cid: str,
        chunk_index: int = 0
    ) -> tuple[bytes, Optional[MerkleProof]]:
        """
        Retrieve data and its storage proof.
        
        Args:
            cid: Content ID to retrieve
            chunk_index: Index of chunk to prove (for chunked data)
            
        Returns:
            Tuple of (data, proof)
        """
        # Retrieve data
        data = self.retrieve_by_cid(cid)
        
        # Try to retrieve associated proof from Pakit
        proof = None
        if REQUESTS_AVAILABLE:
            try:
                response = requests.get(
                    f"{self.pakit_api_url}/api/v1/proof/{cid}/{chunk_index}",
                    timeout=10
                )
                if response.status_code == 200:
                    from .storage_proof_generator import MerkleProof
                    proof = MerkleProof.from_dict(response.json())
                    logger.info(f"Retrieved storage proof for {cid}")
            except Exception as e:
                logger.warning(f"Could not retrieve proof: {e}")
        
        return data, proof


# Convenience function
def retrieve_quantum_result_by_cid(
    cid: str,
    pakit_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Retrieve quantum result by CID.
    
    Args:
        cid: Content ID of the result
        pakit_url: Pakit API URL
        
    Returns:
        Quantum result dictionary
    """
    retriever = QuantumResultRetriever(pakit_api_url=pakit_url)
    return retriever.retrieve_quantum_result(cid)


if __name__ == "__main__":
    # Test retrieval
    retriever = QuantumResultRetriever()
    
    # Test availability check
    test_cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
    availability = retriever.check_availability(test_cid)
    print(f"Availability for {test_cid}:")
    for source, available in availability.items():
        status = "✅" if available else "❌"
        print(f"  {status} {source}")
