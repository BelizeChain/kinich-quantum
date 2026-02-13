"""
DAG Content Addressing for Quantum Results

Implements IPFS-compatible content addressing (CID) for quantum computation results.
Supports both CIDv0 and CIDv1 formats with configurable hash functions.

Integration with Pakit DAG storage for BelizeChain sovereign data storage.
"""

import hashlib
import json
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

try:
    from cid import make_cid, CIDv0, CIDv1
    from multiformats import multibase, multihash, multicodec
    CID_AVAILABLE = True
except ImportError:
    CID_AVAILABLE = False
    logger.warning("py-cid not installed. DAG content addressing limited.")


class HashFunction(Enum):
    """Supported hash functions for content addressing."""
    SHA256 = "sha2-256"
    SHA512 = "sha2-512"
    BLAKE2B_256 = "blake2b-256"
    BLAKE3 = "blake3"


class CIDVersion(Enum):
    """CID version for content addressing."""
    V0 = 0  # Legacy, SHA256 only
    V1 = 1  # Modern, supports multiple codecs


@dataclass
class DAGContent:
    """DAG content with metadata."""
    data: bytes
    cid: str
    size: int
    hash_function: HashFunction
    multihash_bytes: bytes
    links: List[str] = None  # CIDs of linked content (for chunked data)


class DAGContentAddressing:
    """
    Content addressing system for quantum results using IPFS DAG format.
    
    Generates Content IDs (CIDs) for quantum computation results,
    enabling content-addressable storage in Pakit.
    """
    
    def __init__(
        self,
        hash_function: HashFunction = HashFunction.SHA256,
        cid_version: CIDVersion = CIDVersion.V1,
        codec: str = "dag-cbor"
    ):
        """
        Initialize DAG content addressing.
        
        Args:
            hash_function: Hash function to use (default: SHA256)
            cid_version: CID version (V0 or V1, default: V1)
            codec: Multicodec for data format (default: dag-cbor)
        """
        self.hash_function = hash_function
        self.cid_version = cid_version
        self.codec = codec
        
        if not CID_AVAILABLE:
            logger.warning("CID library not available. Using basic hash-based addressing.")
    
    def generate_cid(self, data: bytes) -> str:
        """
        Generate CID for data.
        
        Args:
            data: Raw bytes to address
            
        Returns:
            Content ID (CID) as base58 or base32 string
        """
        if CID_AVAILABLE:
            return self._generate_cid_with_library(data)
        else:
            return self._generate_cid_fallback(data)
    
    def _generate_cid_with_library(self, data: bytes) -> str:
        """Generate CID using py-cid library."""
        # Create multihash
        mh = multihash.digest(data, self.hash_function.value)
        
        # Create CID based on version
        if self.cid_version == CIDVersion.V0:
            # CIDv0: base58btc-encoded multihash (legacy, SHA256 only)
            if self.hash_function != HashFunction.SHA256:
                logger.warning("CIDv0 only supports SHA256. Falling back to SHA256.")
            cid = CIDv0(mh.digest)
        else:
            # CIDv1: multibase-encoded (codec + multihash)
            codec_code = multicodec.get(self.codec)
            cid = CIDv1(codec_code, mh)
        
        return str(cid)
    
    def _generate_cid_fallback(self, data: bytes) -> str:
        """
        Generate CID without py-cid library (basic implementation).
        
        Uses simple hash-based addressing: "bafybeif..." format simulation.
        """
        # Compute hash
        if self.hash_function == HashFunction.SHA256:
            hash_bytes = hashlib.sha256(data).digest()
        elif self.hash_function == HashFunction.SHA512:
            hash_bytes = hashlib.sha512(data).digest()
        elif self.hash_function == HashFunction.BLAKE2B_256:
            hash_bytes = hashlib.blake2b(data, digest_size=32).digest()
        else:
            # Fallback to SHA256
            hash_bytes = hashlib.sha256(data).digest()
        
        # Simple CIDv1-like format: bafybei + base32(hash)
        # This is not a true CID but serves as a content identifier
        if self.cid_version == CIDVersion.V1:
            # Base32 encode (lowercase, no padding)
            import base64
            b32_hash = base64.b32encode(hash_bytes).decode('ascii').lower().rstrip('=')
            return f"bafybei{b32_hash}"
        else:
            # Base58 encode for V0 (simplified)
            import base64
            b58_hash = base64.b58encode(hash_bytes).decode('ascii')
            return f"Qm{b58_hash}"
    
    def generate_dag_content(
        self,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DAGContent:
        """
        Generate DAG content with CID and metadata.
        
        Args:
            data: Raw data bytes
            metadata: Optional metadata to include
            
        Returns:
            DAGContent object with CID and metadata
        """
        # Generate CID
        cid = self.generate_cid(data)
        
        # Compute multihash for proofs
        if self.hash_function == HashFunction.SHA256:
            multihash_bytes = hashlib.sha256(data).digest()
        elif self.hash_function == HashFunction.SHA512:
            multihash_bytes = hashlib.sha512(data).digest()
        elif self.hash_function == HashFunction.BLAKE2B_256:
            multihash_bytes = hashlib.blake2b(data, digest_size=32).digest()
        else:
            multihash_bytes = hashlib.sha256(data).digest()
        
        return DAGContent(
            data=data,
            cid=cid,
            size=len(data),
            hash_function=self.hash_function,
            multihash_bytes=multihash_bytes,
            links=[]
        )
    
    def verify_cid(self, data: bytes, expected_cid: str) -> bool:
        """
        Verify that data matches expected CID.
        
        Args:
            data: Data to verify
            expected_cid: Expected CID
            
        Returns:
            True if data matches CID
        """
        actual_cid = self.generate_cid(data)
        return actual_cid == expected_cid
    
    def chunk_data(
        self,
        data: bytes,
        chunk_size: int = 256 * 1024  # 256KB default
    ) -> List[DAGContent]:
        """
        Chunk large data into smaller DAG blocks.
        
        Args:
            data: Data to chunk
            chunk_size: Size of each chunk in bytes (default: 256KB)
            
        Returns:
            List of DAGContent objects (chunks)
        """
        chunks = []
        offset = 0
        
        while offset < len(data):
            chunk_data = data[offset:offset + chunk_size]
            chunk_content = self.generate_dag_content(chunk_data)
            chunks.append(chunk_content)
            offset += chunk_size
        
        logger.info(f"Chunked {len(data)} bytes into {len(chunks)} chunks")
        return chunks
    
    def create_dag_manifest(
        self,
        chunks: List[DAGContent],
        metadata: Optional[Dict[str, Any]] = None
    ) -> DAGContent:
        """
        Create manifest DAG block linking to all chunks.
        
        Args:
            chunks: List of chunk DAGContent objects
            metadata: Optional metadata for manifest
            
        Returns:
            Manifest DAGContent linking to all chunks
        """
        manifest = {
            'version': '1.0',
            'chunk_count': len(chunks),
            'total_size': sum(c.size for c in chunks),
            'chunks': [
                {
                    'cid': c.cid,
                    'size': c.size,
                    'hash': c.multihash_bytes.hex()
                }
                for c in chunks
            ],
            'metadata': metadata or {}
        }
        
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode('utf-8')
        manifest_content = self.generate_dag_content(manifest_bytes)
        manifest_content.links = [c.cid for c in chunks]
        
        logger.info(f"Created manifest {manifest_content.cid} for {len(chunks)} chunks")
        return manifest_content
    
    def parse_cid(self, cid_str: str) -> Dict[str, Any]:
        """
        Parse CID and extract information.
        
        Args:
            cid_str: CID string to parse
            
        Returns:
            Dictionary with CID information
        """
        if CID_AVAILABLE:
            try:
                cid = make_cid(cid_str)
                return {
                    'version': cid.version,
                    'codec': cid.codec.name if hasattr(cid, 'codec') else 'unknown',
                    'hash_function': cid.multihash.name if hasattr(cid, 'multihash') else 'unknown',
                    'valid': True
                }
            except Exception as e:
                logger.error(f"Failed to parse CID: {e}")
                return {'valid': False, 'error': str(e)}
        else:
            # Basic parsing for fallback CIDs
            if cid_str.startswith('bafybei'):
                return {
                    'version': 1,
                    'codec': 'dag-cbor',
                    'hash_function': self.hash_function.value,
                    'valid': True,
                    'note': 'Fallback CID (py-cid not available)'
                }
            elif cid_str.startswith('Qm'):
                return {
                    'version': 0,
                    'codec': 'dag-pb',
                    'hash_function': 'sha2-256',
                    'valid': True,
                    'note': 'Fallback CID (py-cid not available)'
                }
            else:
                return {'valid': False, 'error': 'Unknown CID format'}


# Convenience functions
def generate_quantum_result_cid(result_data: Dict[str, Any]) -> str:
    """
    Generate CID for quantum result data.
    
    Args:
        result_data: Quantum result dictionary
        
    Returns:
        Content ID (CID)
    """
    dag = DAGContentAddressing()
    result_json = json.dumps(result_data, sort_keys=True).encode('utf-8')
    return dag.generate_cid(result_json)


def verify_quantum_result_integrity(
    result_data: Dict[str, Any],
    expected_cid: str
) -> bool:
    """
    Verify quantum result data matches expected CID.
    
    Args:
        result_data: Quantum result dictionary
        expected_cid: Expected CID
        
    Returns:
        True if data matches CID
    """
    dag = DAGContentAddressing()
    result_json = json.dumps(result_data, sort_keys=True).encode('utf-8')
    return dag.verify_cid(result_json, expected_cid)


if __name__ == "__main__":
    # Test DAG content addressing
    dag = DAGContentAddressing()
    
    # Test basic CID generation
    test_data = b"Hello, Quantum World!"
    cid = dag.generate_cid(test_data)
    print(f"CID: {cid}")
    
    # Test verification
    is_valid = dag.verify_cid(test_data, cid)
    print(f"Verification: {is_valid}")
    
    # Test chunking
    large_data = b"X" * (1024 * 1024)  # 1MB
    chunks = dag.chunk_data(large_data, chunk_size=256*1024)
    print(f"Chunks: {len(chunks)}")
    
    # Test manifest
    manifest = dag.create_dag_manifest(chunks, metadata={'type': 'quantum_result'})
    print(f"Manifest CID: {manifest.cid}")
