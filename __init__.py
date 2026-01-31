"""
Kinich - BelizeChain Distributed Quantum Computing Layer

Named after the Mayan sun god, Kinich provides distributed quantum computing
infrastructure for BelizeChain's sovereign blockchain ecosystem.

Features:
- Multi-backend quantum execution (Azure, SpinQ, IBM, Google)
- Distributed job scheduling across quantum nodes
- Blockchain-integrated Proof of Quantum Work
- Quantum-enhanced cryptography, optimization, and AI
- P2P quantum node network
- Hybrid classical-quantum workload distribution

Supported Backends:
- Azure Quantum (primary production)
- IBM Quantum / Qiskit
- Google Cirq
- SpinQ quantum computers (community nodes)
- Local simulators (development)

Integration with BelizeChain:
- Economy pallet: Transaction optimization
- Governance pallet: Quantum voting algorithms
- Staking pallet: Proof of Quantum Work validation
- Nawal AI: Quantum-enhanced federated learning

Author: BelizeChain Team
License: MIT
"""

from .core import (
    QuantumNode,
    JobScheduler,
    CircuitOptimizer,
    ResultAggregator,
    QuantumJob,
    JobType,
    JobStatus,
    CryptographyJob,
    OptimizationJob,
    SimulationJob,
    AIEnhancementJob,
)

# Adapters commented out temporarily - qiskit API has changed
# from .adapters import (
#     QiskitAdapter,
#     AzureAdapter,
#     SpinQAdapter,
#     AdapterRegistry,
# )

# Jobs are in core now
# from .jobs import (
#     QuantumJob,
#     JobType,
#     JobStatus,
#     CryptographyJob,
#     OptimizationJob,
#     SimulationJob,
#     AIEnhancementJob,
# )

__all__ = [
    # Core
    "QuantumNode",
    "JobScheduler",
    "CircuitOptimizer",
    "ResultAggregator",
    
    # Adapters - temporarily disabled due to qiskit API changes
    # "QiskitAdapter",
    # "AzureAdapter",
    # "SpinQAdapter",
    # "AdapterRegistry",
    
    # Jobs
    "QuantumJob",
    "JobType",
    "JobStatus",
    "CryptographyJob",
    "OptimizationJob",
    "SimulationJob",
    "AIEnhancementJob",
]

__version__ = "0.1.0"
__author__ = "BelizeChain Team"
__license__ = "MIT"

# Kinich configuration
KINICH_VERSION = "0.1.0"
SUPPORTED_BACKENDS = [
    "azure_quantum",
    "ibm_quantum",
    "google_cirq",
    "spinq",
    "qiskit_aer",
]

# Default configuration
DEFAULT_CONFIG = {
    "node": {
        "name": "kinich-node",
        "port": 9950,
        "max_concurrent_jobs": 10,
        "enable_p2p": True,
    },
    "quantum": {
        "default_backend": "qiskit_aer",
        "default_shots": 1024,
        "optimization_level": 2,
        "enable_caching": True,
    },
    "blockchain": {
        "rpc_url": "ws://localhost:9944",
        "submit_results": True,
        "proof_of_quantum_work": True,
    },
    "monitoring": {
        "prometheus_port": 9951,
        "log_level": "INFO",
        "metrics_enabled": True,
    },
}
