"""
Kinich Job Types and Classes

Quantum job types, status enums, and base job classes.
Separated to avoid circular dependencies with quantum_node.

Author: BelizeChain Team
License: MIT
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import json


class JobType(Enum):
    """Types of quantum computing jobs supported by Kinich."""
    
    # Cryptography jobs
    QUANTUM_KEY_GENERATION = "quantum_key_generation"
    POST_QUANTUM_CRYPTO = "post_quantum_crypto"
    QUANTUM_RANDOM = "quantum_random"
    HASH_VERIFICATION = "hash_verification"
    
    # Optimization jobs
    QAOA_OPTIMIZATION = "qaoa_optimization"
    VQE_OPTIMIZATION = "vqe_optimization"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    ROUTE_OPTIMIZATION = "route_optimization"
    
    # Simulation jobs
    QUANTUM_CHEMISTRY = "quantum_chemistry"
    MATERIAL_SIMULATION = "material_simulation"
    MOLECULAR_DYNAMICS = "molecular_dynamics"
    
    # AI/ML enhancement
    QUANTUM_NEURAL_NETWORK = "quantum_neural_network"
    QUANTUM_KERNEL = "quantum_kernel"
    FEDERATED_LEARNING_AGGREGATION = "federated_learning_aggregation"
    VARIATIONAL_CLASSIFIER = "variational_classifier"
    
    # Blockchain consensus
    CONSENSUS_VERIFICATION = "consensus_verification"
    BLOCK_VALIDATION = "block_validation"
    TRANSACTION_ORDERING = "transaction_ordering"
    
    # General computation
    CIRCUIT_EXECUTION = "circuit_execution"
    CUSTOM = "custom"


class JobStatus(Enum):
    """Status of a quantum job."""
    
    PENDING = "pending"
    QUEUED = "queued"
    DISPATCHING = "dispatching"
    RUNNING = "running"
    POST_PROCESSING = "post_processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"


class JobPriority(Enum):
    """Priority levels for job scheduling."""
    
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BATCH = 4


@dataclass
class QuantumJobMetadata:
    """Metadata for quantum job tracking."""
    
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    originator: str = "unknown"
    originating_pallet: Optional[str] = None
    originating_transaction: Optional[str] = None
    
    backend: Optional[str] = None
    node_id: Optional[str] = None
    attempts: int = 0
    
    estimated_qubits: Optional[int] = None
    estimated_depth: Optional[int] = None
    estimated_shots: Optional[int] = None
    actual_execution_time: Optional[float] = None
    
    estimated_cost: Optional[float] = None
    actual_cost: Optional[float] = None
    
    on_chain_submission_tx: Optional[str] = None
    proof_of_quantum_work_score: Optional[float] = None


@dataclass
class QuantumJobResult:
    """Result from quantum job execution."""
    
    job_id: str
    status: JobStatus
    
    counts: Optional[Dict[str, int]] = None
    statevector: Optional[List[complex]] = None
    probabilities: Optional[Dict[str, float]] = None
    
    classical_result: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    
    fidelity: Optional[float] = None
    confidence: Optional[float] = None
    
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    execution_time: float = 0.0
    shots_used: Optional[int] = None
    backend_used: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "counts": self.counts,
            "classical_result": self.classical_result,
            "metadata": self.metadata,
            "fidelity": self.fidelity,
            "confidence": self.confidence,
            "error": self.error,
            "execution_time": self.execution_time,
            "shots_used": self.shots_used,
            "backend_used": self.backend_used,
        }
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class QuantumJob:
    """Base class for quantum computing jobs in Kinich."""
    
    job_type: JobType
    metadata: QuantumJobMetadata = field(default_factory=QuantumJobMetadata)
    
    priority: JobPriority = JobPriority.NORMAL
    max_retries: int = 3
    timeout_seconds: int = 300
    
    num_qubits: Optional[int] = None
    shots: int = 1024
    optimization_level: int = 1
    backend_preference: Optional[List[str]] = None
    
    input_data: Dict[str, Any] = field(default_factory=dict)
    circuit_json: Optional[str] = None
    
    status: JobStatus = JobStatus.PENDING
    result: Optional[QuantumJobResult] = None
    
    on_complete_callback: Optional[str] = None
    on_error_callback: Optional[str] = None
    
    def get_job_id(self) -> str:
        """Get unique job ID."""
        return self.metadata.job_id
    
    def mark_started(self, backend: str, node_id: str) -> None:
        """Mark job as started."""
        self.status = JobStatus.RUNNING
        self.metadata.started_at = datetime.now()
        self.metadata.backend = backend
        self.metadata.node_id = node_id
        self.metadata.attempts += 1
    
    def mark_completed(self, result: QuantumJobResult) -> None:
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        self.result = result
        self.metadata.completed_at = datetime.now()
    
    def mark_failed(self, error: str, error_details: Optional[Dict] = None) -> None:
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.result = QuantumJobResult(
            job_id=self.get_job_id(),
            status=JobStatus.FAILED,
            error=error,
            error_details=error_details,
        )
        self.metadata.completed_at = datetime.now()
    
    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return (
            self.status == JobStatus.FAILED and
            self.metadata.attempts < self.max_retries
        )
    
    def get_execution_time(self) -> Optional[float]:
        """Get total execution time in seconds."""
        if self.metadata.started_at and self.metadata.completed_at:
            delta = self.metadata.completed_at - self.metadata.started_at
            return delta.total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            "job_id": self.get_job_id(),
            "job_type": self.job_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "num_qubits": self.num_qubits,
            "shots": self.shots,
            "input_data": self.input_data,
            "created_at": self.metadata.created_at.isoformat(),
            "started_at": self.metadata.started_at.isoformat() if self.metadata.started_at else None,
            "completed_at": self.metadata.completed_at.isoformat() if self.metadata.completed_at else None,
            "backend": self.metadata.backend,
            "node_id": self.metadata.node_id,
            "attempts": self.metadata.attempts,
        }
    
    def to_json(self) -> str:
        """Convert job to JSON string."""
        return json.dumps(self.to_dict(), default=str)


# Specialized job types

@dataclass
class CryptographyJob(QuantumJob):
    """Quantum cryptography job."""
    
    crypto_operation: str = "key_generation"
    key_length: int = 256
    algorithm: str = "qrng"
    
    def __post_init__(self):
        self.job_type = JobType.QUANTUM_KEY_GENERATION


@dataclass
class OptimizationJob(QuantumJob):
    """Quantum optimization job (QAOA, VQE, etc.)."""
    
    algorithm: str = "QAOA"
    cost_function: Optional[str] = None
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    initial_parameters: Optional[List[float]] = None
    max_iterations: int = 100
    
    def __post_init__(self):
        self.job_type = JobType.QAOA_OPTIMIZATION


@dataclass
class SimulationJob(QuantumJob):
    """Quantum simulation job (chemistry, materials, etc.)."""
    
    simulation_type: str = "chemistry"
    molecule: Optional[str] = None
    hamiltonian: Optional[str] = None
    basis_set: str = "sto-3g"
    
    def __post_init__(self):
        self.job_type = JobType.QUANTUM_CHEMISTRY


@dataclass
class AIEnhancementJob(QuantumJob):
    """Quantum-enhanced AI/ML job."""
    
    ai_operation: str = "kernel"
    model_parameters: Optional[List[float]] = None
    feature_dimension: Optional[int] = None
    training_data: Optional[List[List[float]]] = None
    
    def __post_init__(self):
        self.job_type = JobType.QUANTUM_NEURAL_NETWORK


@dataclass
class ConsensusJob(QuantumJob):
    """Blockchain consensus verification job."""
    
    consensus_operation: str = "verification"
    validator_votes: Optional[List[bool]] = None
    block_data: Optional[str] = None
    threshold: float = 0.67
    
    def __post_init__(self):
        self.job_type = JobType.CONSENSUS_VERIFICATION
