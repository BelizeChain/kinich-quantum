"""
Tests for Kinich core job types and lifecycle management.

Tests job creation, status transitions, metadata tracking,
serialization, retry logic, and execution time calculation.

Author: BelizeChain Team
License: MIT
"""

import pytest
from datetime import datetime, timedelta
import json

from kinich.core import (
    QuantumJob,
    QuantumJobMetadata,
    QuantumJobResult,
    JobType,
    JobStatus,
    JobPriority,
    CryptographyJob,
    OptimizationJob,
    SimulationJob,
    AIEnhancementJob,
    ConsensusJob,
)


@pytest.mark.unit
class TestQuantumJob:
    """Test base QuantumJob class."""
    
    def test_job_creation(self, simple_quantum_job):
        """Test creating a basic quantum job."""
        assert simple_quantum_job.job_type == JobType.CIRCUIT_EXECUTION
        assert simple_quantum_job.num_qubits == 2
        assert simple_quantum_job.shots == 1024
        assert simple_quantum_job.status == JobStatus.PENDING
        assert simple_quantum_job.priority == JobPriority.NORMAL
    
    def test_job_id_unique(self):
        """Test that each job gets a unique ID."""
        job1 = QuantumJob(job_type=JobType.CIRCUIT_EXECUTION)
        job2 = QuantumJob(job_type=JobType.CIRCUIT_EXECUTION)
        assert job1.get_job_id() != job2.get_job_id()
    
    def test_mark_started(self, simple_quantum_job):
        """Test marking job as started."""
        backend = "qiskit_aer"
        node_id = "node-001"
        
        simple_quantum_job.mark_started(backend, node_id)
        
        assert simple_quantum_job.status == JobStatus.RUNNING
        assert simple_quantum_job.metadata.backend == backend
        assert simple_quantum_job.metadata.node_id == node_id
        assert simple_quantum_job.metadata.attempts == 1
        assert simple_quantum_job.metadata.started_at is not None
    
    def test_mark_completed(self, simple_quantum_job, completed_job_result):
        """Test marking job as completed."""
        simple_quantum_job.mark_completed(completed_job_result)
        
        assert simple_quantum_job.status == JobStatus.COMPLETED
        assert simple_quantum_job.result == completed_job_result
        assert simple_quantum_job.metadata.completed_at is not None
    
    def test_mark_failed(self, simple_quantum_job):
        """Test marking job as failed."""
        error_msg = "Backend timeout"
        error_details = {"timeout": 300}
        
        simple_quantum_job.mark_failed(error_msg, error_details)
        
        assert simple_quantum_job.status == JobStatus.FAILED
        assert simple_quantum_job.result.error == error_msg
        assert simple_quantum_job.result.error_details == error_details
        assert simple_quantum_job.metadata.completed_at is not None
    
    def test_can_retry_when_failed_below_max(self, simple_quantum_job):
        """Test retry is allowed when job failed and attempts < max_retries."""
        simple_quantum_job.max_retries = 3
        simple_quantum_job.metadata.attempts = 2
        simple_quantum_job.mark_failed("Test failure")
        
        assert simple_quantum_job.can_retry() is True
    
    def test_cannot_retry_when_max_attempts_reached(self, simple_quantum_job):
        """Test retry is not allowed when max attempts reached."""
        simple_quantum_job.max_retries = 3
        simple_quantum_job.metadata.attempts = 3
        simple_quantum_job.mark_failed("Test failure")
        
        assert simple_quantum_job.can_retry() is False
    
    def test_cannot_retry_when_completed(self, simple_quantum_job, completed_job_result):
        """Test retry is not allowed for completed jobs."""
        simple_quantum_job.mark_completed(completed_job_result)
        
        assert simple_quantum_job.can_retry() is False
    
    def test_execution_time_calculation(self, simple_quantum_job):
        """Test execution time calculation."""
        simple_quantum_job.metadata.started_at = datetime.now()
        simple_quantum_job.metadata.completed_at = datetime.now() + timedelta(seconds=5.5)
        
        execution_time = simple_quantum_job.get_execution_time()
        assert execution_time is not None
        assert 5.0 <= execution_time <= 6.0  # Allow some margin for timing
    
    def test_execution_time_none_when_not_completed(self, simple_quantum_job):
        """Test execution time is None when job not completed."""
        assert simple_quantum_job.get_execution_time() is None
    
    def test_job_to_dict_serialization(self, simple_quantum_job):
        """Test job serialization to dictionary."""
        job_dict = simple_quantum_job.to_dict()
        
        assert job_dict["job_type"] == JobType.CIRCUIT_EXECUTION.value
        assert job_dict["status"] == JobStatus.PENDING.value
        assert job_dict["priority"] == JobPriority.NORMAL.value
        assert job_dict["num_qubits"] == 2
        assert job_dict["shots"] == 1024
        assert "job_id" in job_dict
        assert "created_at" in job_dict
    
    def test_job_to_json_serialization(self, simple_quantum_job):
        """Test job serialization to JSON string."""
        job_json = simple_quantum_job.to_json()
        
        assert isinstance(job_json, str)
        parsed = json.loads(job_json)
        assert parsed["job_type"] == JobType.CIRCUIT_EXECUTION.value
        assert parsed["num_qubits"] == 2


@pytest.mark.unit
class TestCryptographyJob:
    """Test CryptographyJob class."""
    
    def test_cryptography_job_creation(self, cryptography_job):
        """Test creating cryptography job."""
        assert cryptography_job.job_type == JobType.QUANTUM_KEY_GENERATION
        assert cryptography_job.crypto_operation == "key_generation"
        assert cryptography_job.key_length == 256
        assert cryptography_job.algorithm == "qrng"
        assert cryptography_job.shots == 2048
    
    def test_cryptography_job_inherits_base_functionality(self, cryptography_job):
        """Test that cryptography job inherits base job functionality."""
        assert cryptography_job.status == JobStatus.PENDING
        assert cryptography_job.priority == JobPriority.NORMAL
        assert cryptography_job.get_job_id() is not None
        
        # Test status transition
        cryptography_job.mark_started("qiskit_aer", "node-001")
        assert cryptography_job.status == JobStatus.RUNNING


@pytest.mark.unit
class TestOptimizationJob:
    """Test OptimizationJob class."""
    
    def test_optimization_job_creation(self, optimization_job):
        """Test creating optimization job."""
        assert optimization_job.job_type == JobType.QAOA_OPTIMIZATION
        assert optimization_job.algorithm == "QAOA"
        assert optimization_job.num_qubits == 4
        assert optimization_job.cost_function == "max_cut"
        assert optimization_job.max_iterations == 100
    
    def test_optimization_job_with_constraints(self):
        """Test optimization job with constraints."""
        job = OptimizationJob(
            job_type=JobType.QAOA_OPTIMIZATION,
            algorithm="QAOA",
            num_qubits=4,
            constraints=[
                {"type": "inequality", "expression": "x1 + x2 <= 10"},
                {"type": "equality", "expression": "x3 == 5"}
            ],
        )
        
        assert len(job.constraints) == 2
        assert job.constraints[0]["type"] == "inequality"


@pytest.mark.unit
class TestSimulationJob:
    """Test SimulationJob class."""
    
    def test_simulation_job_creation(self, simulation_job):
        """Test creating simulation job."""
        assert simulation_job.job_type == JobType.QUANTUM_CHEMISTRY
        assert simulation_job.simulation_type == "chemistry"
        assert simulation_job.molecule == "H2"
        assert simulation_job.hamiltonian == "pauli_z"
        assert simulation_job.basis_set == "sto-3g"
    
    def test_simulation_job_materials_type(self):
        """Test materials simulation job."""
        job = SimulationJob(
            job_type=JobType.MATERIAL_SIMULATION,
            simulation_type="materials",
            molecule="Si",
            num_qubits=8,
        )
        
        assert job.simulation_type == "materials"
        assert job.molecule == "Si"


@pytest.mark.unit
class TestAIEnhancementJob:
    """Test AIEnhancementJob class."""
    
    def test_ai_enhancement_job_creation(self, ai_enhancement_job):
        """Test creating AI enhancement job."""
        assert ai_enhancement_job.job_type == JobType.QUANTUM_NEURAL_NETWORK
        assert ai_enhancement_job.ai_operation == "kernel"
        assert ai_enhancement_job.feature_dimension == 4
        assert ai_enhancement_job.num_qubits == 4
        assert len(ai_enhancement_job.training_data) == 2
    
    def test_ai_enhancement_job_with_model_parameters(self):
        """Test AI job with model parameters."""
        job = AIEnhancementJob(
            job_type=JobType.QUANTUM_KERNEL,
            ai_operation="kernel",
            model_parameters=[0.1, 0.2, 0.3, 0.4],
            feature_dimension=4,
            num_qubits=4,
        )
        
        assert job.model_parameters is not None
        assert len(job.model_parameters) == 4
        assert job.model_parameters[0] == 0.1


@pytest.mark.unit
class TestConsensusJob:
    """Test ConsensusJob class."""
    
    def test_consensus_job_creation(self, consensus_job):
        """Test creating consensus job."""
        assert consensus_job.job_type == JobType.CONSENSUS_VERIFICATION
        assert consensus_job.consensus_operation == "verification"
        assert len(consensus_job.validator_votes) == 4
        assert consensus_job.block_data == "0xabc123"
        assert consensus_job.threshold == 0.67
    
    def test_consensus_job_voting_operation(self):
        """Test consensus job for voting."""
        job = ConsensusJob(
            job_type=JobType.CONSENSUS_VERIFICATION,
            consensus_operation="voting",
            validator_votes=[True, True, True, False, False],
            threshold=0.6,
            num_qubits=5,
        )
        
        assert job.consensus_operation == "voting"
        assert job.validator_votes is not None
        assert len(job.validator_votes) == 5
        assert job.threshold == 0.6


@pytest.mark.unit
class TestQuantumJobResult:
    """Test QuantumJobResult class."""
    
    def test_result_creation(self, completed_job_result):
        """Test creating job result."""
        assert completed_job_result.status == JobStatus.COMPLETED
        assert completed_job_result.counts is not None
        assert completed_job_result.execution_time == 1.5
        assert completed_job_result.backend_used == "qiskit_aer"
    
    def test_result_to_dict(self, completed_job_result):
        """Test result serialization to dict."""
        result_dict = completed_job_result.to_dict()
        
        assert result_dict["status"] == JobStatus.COMPLETED.value
        assert result_dict["counts"] == completed_job_result.counts
        assert result_dict["execution_time"] == 1.5
        assert result_dict["fidelity"] == 0.95
    
    def test_result_to_json(self, completed_job_result):
        """Test result serialization to JSON."""
        result_json = completed_job_result.to_json()
        
        assert isinstance(result_json, str)
        parsed = json.loads(result_json)
        assert parsed["status"] == JobStatus.COMPLETED.value
        assert parsed["backend_used"] == "qiskit_aer"
    
    def test_failed_result(self, failed_job_result):
        """Test failed job result."""
        assert failed_job_result.status == JobStatus.FAILED
        assert failed_job_result.error == "Backend timeout"
        assert failed_job_result.error_details["timeout_seconds"] == 300
        assert failed_job_result.counts is None


@pytest.mark.unit
class TestJobMetadata:
    """Test QuantumJobMetadata class."""
    
    def test_metadata_creation(self, job_metadata):
        """Test creating job metadata."""
        assert job_metadata.job_id == "meta-job-001"
        assert job_metadata.originator == "test-user"
        assert job_metadata.originating_pallet == "staking"
        assert job_metadata.estimated_qubits == 4
        assert job_metadata.estimated_shots == 1024
    
    def test_metadata_timestamps(self):
        """Test metadata timestamp tracking."""
        metadata = QuantumJobMetadata()
        
        assert metadata.created_at is not None
        assert metadata.started_at is None
        assert metadata.completed_at is None
        
        # Simulate job lifecycle
        metadata.started_at = datetime.now()
        assert metadata.started_at is not None
        
        metadata.completed_at = datetime.now()
        assert metadata.completed_at is not None
    
    def test_metadata_cost_tracking(self):
        """Test cost tracking in metadata."""
        metadata = QuantumJobMetadata(
            estimated_cost=0.50,
            actual_cost=0.48,
        )
        
        assert metadata.estimated_cost == 0.50
        assert metadata.actual_cost == 0.48
    
    def test_metadata_blockchain_integration(self):
        """Test blockchain integration fields."""
        metadata = QuantumJobMetadata(
            originating_transaction="0xabc123def456",
            on_chain_submission_tx="0x789ghi012jkl",
            proof_of_quantum_work_score=95.5,
        )
        
        assert metadata.originating_transaction == "0xabc123def456"
        assert metadata.on_chain_submission_tx == "0x789ghi012jkl"
        assert metadata.proof_of_quantum_work_score == 95.5
