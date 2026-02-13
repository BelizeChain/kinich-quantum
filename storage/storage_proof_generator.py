"""
Storage Proof Generator for Quantum Results

Implements Merkle tree-based storage proofs for verifying quantum results
stored in Pakit DAG. Supports proof-of-storage and proof-of-replication.

Integration with BelizeChain for on-chain storage verification.
"""

import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class ProofType(Enum):
    """Type of storage proof."""
    MERKLE_INCLUSION = "merkle_inclusion"  # Prove data is in tree
    MERKLE_RANGE = "merkle_range"  # Prove range of data
    PROOF_OF_REPLICATION = "proof_of_replication"  # Prove data is replicated
    CHALLENGE_RESPONSE = "challenge_response"  # Prove data can be retrieved


@dataclass
class MerkleNode:
    """Node in Merkle tree."""
    hash: bytes
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    is_leaf: bool = False
    data: Optional[bytes] = None
    index: int = -1


@dataclass
class MerkleProof:
    """
    Merkle proof for data inclusion.
    
    Contains the path from leaf to root with sibling hashes.
    """
    leaf_hash: bytes
    leaf_index: int
    root_hash: bytes
    proof_path: List[Tuple[bytes, bool]]  # (hash, is_right_sibling)
    tree_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert proof to dictionary for serialization."""
        return {
            'leaf_hash': self.leaf_hash.hex(),
            'leaf_index': self.leaf_index,
            'root_hash': self.root_hash.hex(),
            'proof_path': [
                {'hash': h.hex(), 'is_right': is_right}
                for h, is_right in self.proof_path
            ],
            'tree_size': self.tree_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MerkleProof':
        """Create proof from dictionary."""
        return cls(
            leaf_hash=bytes.fromhex(data['leaf_hash']),
            leaf_index=data['leaf_index'],
            root_hash=bytes.fromhex(data['root_hash']),
            proof_path=[
                (bytes.fromhex(item['hash']), item['is_right'])
                for item in data['proof_path']
            ],
            tree_size=data['tree_size']
        )


@dataclass
class ReplicationProof:
    """
    Proof of data replication.
    
    Proves that data has been replicated correctly across storage nodes.
    """
    replica_id: int
    challenge: bytes
    response: bytes
    merkle_proof: MerkleProof
    timestamp: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'replica_id': self.replica_id,
            'challenge': self.challenge.hex(),
            'response': self.response.hex(),
            'merkle_proof': self.merkle_proof.to_dict(),
            'timestamp': self.timestamp
        }


class MerkleTree:
    """
    Merkle tree implementation for storage proofs.
    
    Builds a binary tree of hashes for efficient proof generation.
    """
    
    def __init__(self, hash_function: str = "sha256"):
        """
        Initialize Merkle tree.
        
        Args:
            hash_function: Hash function to use (sha256, sha512, blake2b)
        """
        self.hash_function = hash_function
        self.root: Optional[MerkleNode] = None
        self.leaves: List[MerkleNode] = []
        self.tree_height = 0
    
    def _hash(self, data: bytes) -> bytes:
        """Compute hash of data."""
        if self.hash_function == "sha256":
            return hashlib.sha256(data).digest()
        elif self.hash_function == "sha512":
            return hashlib.sha512(data).digest()
        elif self.hash_function == "blake2b":
            return hashlib.blake2b(data, digest_size=32).digest()
        else:
            return hashlib.sha256(data).digest()
    
    def build_tree(self, data_chunks: List[bytes]) -> bytes:
        """
        Build Merkle tree from data chunks.
        
        Args:
            data_chunks: List of data chunks (leaves)
            
        Returns:
            Root hash of the tree
        """
        if not data_chunks:
            raise ValueError("Cannot build tree from empty data")
        
        # Create leaf nodes
        self.leaves = []
        for i, chunk in enumerate(data_chunks):
            leaf_hash = self._hash(chunk)
            leaf = MerkleNode(
                hash=leaf_hash,
                is_leaf=True,
                data=chunk,
                index=i
            )
            self.leaves.append(leaf)
        
        # Build tree bottom-up
        current_level = self.leaves[:]
        height = 0
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                
                # Handle odd number of nodes (duplicate last node)
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    right = left
                
                # Combine hashes
                combined = left.hash + right.hash
                parent_hash = self._hash(combined)
                
                parent = MerkleNode(
                    hash=parent_hash,
                    left=left,
                    right=right,
                    is_leaf=False
                )
                next_level.append(parent)
            
            current_level = next_level
            height += 1
        
        self.root = current_level[0]
        self.tree_height = height
        
        logger.info(f"Built Merkle tree: {len(self.leaves)} leaves, height {height}")
        return self.root.hash
    
    def generate_proof(self, leaf_index: int) -> MerkleProof:
        """
        Generate Merkle proof for a specific leaf.
        
        Args:
            leaf_index: Index of the leaf to prove
            
        Returns:
            MerkleProof object
        """
        if not self.root:
            raise ValueError("Tree not built yet")
        
        if leaf_index < 0 or leaf_index >= len(self.leaves):
            raise ValueError(f"Invalid leaf index: {leaf_index}")
        
        leaf = self.leaves[leaf_index]
        proof_path = []
        
        # Traverse from leaf to root, collecting sibling hashes
        current_level = self.leaves[:]
        current_index = leaf_index
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                
                # Record sibling hash
                if current_index == i:
                    # Current node is left, sibling is right
                    proof_path.append((right.hash, True))
                    current_index = i // 2
                elif current_index == i + 1:
                    # Current node is right, sibling is left
                    proof_path.append((left.hash, False))
                    current_index = i // 2
                
                # Build parent
                combined = left.hash + right.hash
                parent_hash = self._hash(combined)
                parent = MerkleNode(hash=parent_hash, left=left, right=right)
                next_level.append(parent)
            
            current_level = next_level
        
        return MerkleProof(
            leaf_hash=leaf.hash,
            leaf_index=leaf_index,
            root_hash=self.root.hash,
            proof_path=proof_path,
            tree_size=len(self.leaves)
        )
    
    def verify_proof(self, proof: MerkleProof) -> bool:
        """
        Verify a Merkle proof.
        
        Args:
            proof: MerkleProof to verify
            
        Returns:
            True if proof is valid
        """
        # Start with leaf hash
        current_hash = proof.leaf_hash
        
        # Traverse up the tree using proof path
        for sibling_hash, is_right_sibling in proof.proof_path:
            if is_right_sibling:
                # Sibling is on the right
                combined = current_hash + sibling_hash
            else:
                # Sibling is on the left
                combined = sibling_hash + current_hash
            
            current_hash = self._hash(combined)
        
        # Check if we arrived at the root
        return current_hash == proof.root_hash
    
    def get_root_hash(self) -> Optional[bytes]:
        """Get root hash of the tree."""
        return self.root.hash if self.root else None


class StorageProofGenerator:
    """
    Generate and verify storage proofs for quantum results.
    
    Supports multiple proof types including Merkle inclusion proofs
    and proof-of-replication.
    """
    
    def __init__(self, hash_function: str = "sha256"):
        """
        Initialize storage proof generator.
        
        Args:
            hash_function: Hash function to use
        """
        self.hash_function = hash_function
    
    def generate_inclusion_proof(
        self,
        data_chunks: List[bytes],
        chunk_index: int
    ) -> MerkleProof:
        """
        Generate proof that a chunk is included in the dataset.
        
        Args:
            data_chunks: All data chunks
            chunk_index: Index of chunk to prove
            
        Returns:
            MerkleProof for the chunk
        """
        tree = MerkleTree(self.hash_function)
        tree.build_tree(data_chunks)
        return tree.generate_proof(chunk_index)
    
    def verify_inclusion_proof(self, proof: MerkleProof) -> bool:
        """
        Verify inclusion proof.
        
        Args:
            proof: MerkleProof to verify
            
        Returns:
            True if proof is valid
        """
        tree = MerkleTree(self.hash_function)
        return tree.verify_proof(proof)
    
    def generate_replication_proof(
        self,
        data_chunks: List[bytes],
        replica_id: int,
        challenge: bytes
    ) -> ReplicationProof:
        """
        Generate proof of replication.
        
        Proves that data has been replicated and can be retrieved.
        
        Args:
            data_chunks: Replicated data chunks
            replica_id: ID of the replica
            challenge: Random challenge bytes
            
        Returns:
            ReplicationProof
        """
        import time
        
        # Build Merkle tree
        tree = MerkleTree(self.hash_function)
        root_hash = tree.build_tree(data_chunks)
        
        # Generate challenge response (hash of challenge + replica_id + root_hash)
        response_data = challenge + replica_id.to_bytes(4, 'big') + root_hash
        response = hashlib.sha256(response_data).digest()
        
        # Generate Merkle proof for a random chunk (deterministic from challenge)
        chunk_index = int.from_bytes(challenge[:4], 'big') % len(data_chunks)
        merkle_proof = tree.generate_proof(chunk_index)
        
        return ReplicationProof(
            replica_id=replica_id,
            challenge=challenge,
            response=response,
            merkle_proof=merkle_proof,
            timestamp=int(time.time())
        )
    
    def verify_replication_proof(
        self,
        proof: ReplicationProof,
        expected_root_hash: bytes
    ) -> bool:
        """
        Verify proof of replication.
        
        Args:
            proof: ReplicationProof to verify
            expected_root_hash: Expected Merkle root hash
            
        Returns:
            True if proof is valid
        """
        # Verify Merkle proof
        if proof.merkle_proof.root_hash != expected_root_hash:
            logger.error("Root hash mismatch in replication proof")
            return False
        
        tree = MerkleTree(self.hash_function)
        if not tree.verify_proof(proof.merkle_proof):
            logger.error("Invalid Merkle proof in replication proof")
            return False
        
        # Verify challenge response
        response_data = (
            proof.challenge + 
            proof.replica_id.to_bytes(4, 'big') + 
            expected_root_hash
        )
        expected_response = hashlib.sha256(response_data).digest()
        
        if proof.response != expected_response:
            logger.error("Invalid challenge response in replication proof")
            return False
        
        return True
    
    def generate_storage_attestation(
        self,
        cid: str,
        data_chunks: List[bytes],
        replica_id: int = 0
    ) -> Dict[str, Any]:
        """
        Generate complete storage attestation for blockchain submission.
        
        Args:
            cid: Content ID of the data
            data_chunks: Data chunks
            replica_id: Replica ID (default: 0 for original)
            
        Returns:
            Dictionary with attestation data
        """
        # Build Merkle tree
        tree = MerkleTree(self.hash_function)
        root_hash = tree.build_tree(data_chunks)
        
        # Generate proof for first chunk (as example)
        proof = tree.generate_proof(0)
        
        return {
            'cid': cid,
            'merkle_root': root_hash.hex(),
            'chunk_count': len(data_chunks),
            'total_size': sum(len(chunk) for chunk in data_chunks),
            'replica_id': replica_id,
            'proof_sample': proof.to_dict(),
            'hash_function': self.hash_function
        }


# Convenience functions for quantum results
def generate_quantum_result_proof(
    result_cid: str,
    result_chunks: List[bytes],
    chunk_index: int = 0
) -> MerkleProof:
    """
    Generate storage proof for quantum result.
    
    Args:
        result_cid: CID of the quantum result
        result_chunks: Data chunks of the result
        chunk_index: Index of chunk to prove (default: 0)
        
    Returns:
        MerkleProof
    """
    generator = StorageProofGenerator()
    return generator.generate_inclusion_proof(result_chunks, chunk_index)


if __name__ == "__main__":
    # Test storage proof generation
    print("Testing storage proof generator...")
    
    # Create test data
    test_chunks = [
        b"quantum_result_chunk_0",
        b"quantum_result_chunk_1",
        b"quantum_result_chunk_2",
        b"quantum_result_chunk_3"
    ]
    
    # Test Merkle tree
    generator = StorageProofGenerator()
    proof = generator.generate_inclusion_proof(test_chunks, 1)
    print(f"Generated proof for chunk 1")
    print(f"Root hash: {proof.root_hash.hex()}")
    print(f"Proof path length: {len(proof.proof_path)}")
    
    # Verify proof
    is_valid = generator.verify_inclusion_proof(proof)
    print(f"Proof verification: {'✅ VALID' if is_valid else '❌ INVALID'}")
    
    # Test replication proof
    import secrets
    challenge = secrets.token_bytes(32)
    rep_proof = generator.generate_replication_proof(test_chunks, replica_id=1, challenge=challenge)
    print(f"\nGenerated replication proof")
    print(f"Replica ID: {rep_proof.replica_id}")
    
    # Verify replication proof
    is_valid = generator.verify_replication_proof(rep_proof, rep_proof.merkle_proof.root_hash)
    print(f"Replication proof verification: {'✅ VALID' if is_valid else '❌ INVALID'}")
