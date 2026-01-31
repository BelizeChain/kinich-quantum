"""
Pytest configuration and fixtures for Kinich tests.

Provides reusable fixtures for mocking quantum backends,
jobs, nodes, and circuits.

Author: BelizeChain Team
License: MIT
"""

import pytest
from typing import Dict, List, Any, Optional
from datetime import datetime
from unittest.mock import Mock, MagicMock
import uuid

# Import Kinich components
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

# Import NodeConfig and NodeStatus separately to avoid adapter imports
from kinich.core.quantum_node import NodeConfig, NodeStatus


@pytest.fixture
def mock_adapter_registry():
    """Mock adapter registry for quantum backends."""
    registry = Mock()
    registry.get_adapter = Mock(return_value=Mock())
    registry.list_adapters = Mock(return_value=["qiskit_aer", "azure_quantum"])
    registry.register_adapter = Mock()
    return registry


@pytest.fixture
def mock_qiskit_adapter():
    """Mock Qiskit adapter."""
    adapter = Mock()
    adapter.name = "qiskit_aer"
    adapter.is_available = Mock(return_value=True)
    adapter.execute_job = Mock(return_value=QuantumJobResult(
        job_id="test-job-id",
        status=JobStatus.COMPLETED,
        counts={"00": 512, "11": 512},
        execution_time=0.5,
        shots_used=1024,
        backend_used="qiskit_aer",
    ))
    return adapter


@pytest.fixture
def mock_azure_adapter():
    """Mock Azure Quantum adapter."""
    adapter = Mock()
    adapter.name = "azure_quantum"
    adapter.is_available = Mock(return_value=True)
    adapter.execute_job = Mock(return_value=QuantumJobResult(
        job_id="test-job-id",
        status=JobStatus.COMPLETED,
        counts={"00": 256, "11": 768},
        execution_time=1.2,
        shots_used=1024,
        backend_used="azure_quantum",
    ))
    return adapter


@pytest.fixture
def simple_quantum_job():
    """Simple quantum job for testing."""
    return QuantumJob(
        job_type=JobType.CIRCUIT_EXECUTION,
        num_qubits=2,
        shots=1024,
        input_data={"circuit": "simple"},
    )


@pytest.fixture
def cryptography_job():
    """Cryptography job for testing."""
    job = CryptographyJob(
        job_type=JobType.QUANTUM_KEY_GENERATION,
        crypto_operation="key_generation",
        key_length=256,
        algorithm="qrng",
        shots=2048,
    )
    return job


@pytest.fixture
def optimization_job():
    """Optimization job for testing."""
    job = OptimizationJob(
        job_type=JobType.QAOA_OPTIMIZATION,
        algorithm="QAOA",
        num_qubits=4,
        cost_function="max_cut",
        max_iterations=100,
        shots=1024,
    )
    return job


@pytest.fixture
def simulation_job():
    """Simulation job for testing."""
    job = SimulationJob(
        job_type=JobType.QUANTUM_CHEMISTRY,
        simulation_type="chemistry",
        molecule="H2",
        hamiltonian="pauli_z",
        basis_set="sto-3g",
        num_qubits=2,
    )
    return job


@pytest.fixture
def ai_enhancement_job():
    """AI enhancement job for testing."""
    job = AIEnhancementJob(
        job_type=JobType.QUANTUM_NEURAL_NETWORK,
        ai_operation="kernel",
        feature_dimension=4,
        num_qubits=4,
        training_data=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
    )
    return job


@pytest.fixture
def consensus_job():
    """Consensus job for testing."""
    job = ConsensusJob(
        job_type=JobType.CONSENSUS_VERIFICATION,
        consensus_operation="verification",
        validator_votes=[True, True, False, True],
        block_data="0xabc123",
        threshold=0.67,
        num_qubits=4,
    )
    return job


@pytest.fixture
def node_config():
    """Standard node configuration."""
    return NodeConfig(
        node_id="test-node-001",
        node_name="Test Kinich Node",
        rpc_url="ws://localhost:9944",
        max_concurrent_jobs=5,
        max_queue_size=50,
        enable_p2p=False,  # Disable for testing
        enable_metrics=False,  # Disable for testing
    )


@pytest.fixture
def quantum_node(node_config, mock_adapter_registry):
    """Quantum node instance for testing."""
    # Import here to avoid loading adapters at module level
    from kinich.core.quantum_node import QuantumNode
    node = QuantumNode(config=node_config, adapter_registry=mock_adapter_registry)
    node.status = NodeStatus.READY
    return node


@pytest.fixture
def completed_job_result():
    """Completed quantum job result."""
    return QuantumJobResult(
        job_id="completed-job-001",
        status=JobStatus.COMPLETED,
        counts={"00": 400, "01": 200, "10": 200, "11": 224},
        probabilities={"00": 0.39, "01": 0.20, "10": 0.20, "11": 0.21},
        execution_time=1.5,
        shots_used=1024,
        backend_used="qiskit_aer",
        fidelity=0.95,
        confidence=0.89,
    )


@pytest.fixture
def failed_job_result():
    """Failed quantum job result."""
    return QuantumJobResult(
        job_id="failed-job-001",
        status=JobStatus.FAILED,
        error="Backend timeout",
        error_details={"timeout_seconds": 300, "backend": "azure_quantum"},
        execution_time=300.0,
    )


@pytest.fixture
def mock_quantum_circuit():
    """Mock quantum circuit."""
    circuit = Mock()
    circuit.num_qubits = 4
    circuit.depth = Mock(return_value=10)
    circuit.num_gates = Mock(return_value=25)
    circuit.to_json = Mock(return_value='{"qubits": 4, "gates": []}')
    return circuit


@pytest.fixture
def job_metadata():
    """Standard job metadata."""
    return QuantumJobMetadata(
        job_id="meta-job-001",
        created_at=datetime.now(),
        originator="test-user",
        originating_pallet="staking",
        estimated_qubits=4,
        estimated_shots=1024,
    )


@pytest.fixture
def multiple_jobs(simple_quantum_job, cryptography_job, optimization_job):
    """List of multiple jobs for queue testing."""
    return [
        simple_quantum_job,
        cryptography_job,
        optimization_job,
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_backend: mark test as requiring real quantum backend"
    )
