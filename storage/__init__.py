"""
Kinich Storage Integration

Enhanced DAG-based storage with proofs for quantum computation results.
"""

from kinich.storage.quantum_results_store import QuantumResultsStore
from kinich.storage.dag_content_addressing import (
    DAGContentAddressing,
    DAGContent,
    HashFunction,
    CIDVersion,
    generate_quantum_result_cid,
    verify_quantum_result_integrity
)
from kinich.storage.storage_proof_generator import (
    StorageProofGenerator,
    MerkleProof,
    MerkleTree,
    ReplicationProof,
    ProofType,
    generate_quantum_result_proof
)
from kinich.storage.result_retriever import (
    QuantumResultRetriever,
    ResultRetrievalError,
    retrieve_quantum_result_by_cid
)

__all__ = [
    # Main store
    'QuantumResultsStore',
    
    # DAG content addressing
    'DAGContentAddressing',
    'DAGContent',
    'HashFunction',
    'CIDVersion',
    'generate_quantum_result_cid',
    'verify_quantum_result_integrity',
    
    # Storage proofs
    'StorageProofGenerator',
    'MerkleProof',
    'MerkleTree',
    'ReplicationProof',
    'ProofType',
    'generate_quantum_result_proof',
    
    # Result retrieval
    'QuantumResultRetriever',
    'ResultRetrievalError',
    'retrieve_quantum_result_by_cid',
]
