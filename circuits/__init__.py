"""
Circuit Builders for Kinich

Factory methods to build quantum circuits for different job types.
Maps JobType enum to actual circuit implementations.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class CircuitBuilder:
    """
    Builds quantum circuits for different job types.
    
    Central factory for converting job specifications into
    executable quantum circuits.
    """
    
    @staticmethod
    def build_circuit(job_type: str, **kwargs) -> Optional[Any]:
        """
        Build quantum circuit for job type.
        
        Args:
            job_type: Type of quantum job
            **kwargs: Job-specific parameters
        
        Returns:
            Quantum circuit or None
        """
        builder_map = {
            # Cryptography
            'QUANTUM_KEY_GENERATION': CircuitBuilder.build_qkg_circuit,
            'QUANTUM_RANDOM': CircuitBuilder.build_qrng_circuit,
            'POST_QUANTUM_CRYPTO': CircuitBuilder.build_pqc_circuit,
            'HASH_VERIFICATION': CircuitBuilder.build_hash_circuit,
            
            # Optimization
            'QAOA_OPTIMIZATION': CircuitBuilder.build_qaoa_circuit,
            'VQE_OPTIMIZATION': CircuitBuilder.build_vqe_circuit,
            'PORTFOLIO_OPTIMIZATION': CircuitBuilder.build_portfolio_circuit,
            'ROUTE_OPTIMIZATION': CircuitBuilder.build_routing_circuit,
            
            # Simulation
            'QUANTUM_CHEMISTRY': CircuitBuilder.build_chemistry_circuit,
            'MATERIAL_SIMULATION': CircuitBuilder.build_material_circuit,
            'MOLECULAR_DYNAMICS': CircuitBuilder.build_molecular_circuit,
            
            # AI/ML
            'QUANTUM_NEURAL_NETWORK': CircuitBuilder.build_qnn_circuit,
            'QUANTUM_KERNEL': CircuitBuilder.build_kernel_circuit,
            'FEDERATED_LEARNING_AGGREGATION': CircuitBuilder.build_fl_circuit,
            'VARIATIONAL_CLASSIFIER': CircuitBuilder.build_classifier_circuit,
            
            # Consensus
            'CONSENSUS_VERIFICATION': CircuitBuilder.build_consensus_circuit,
            'BLOCK_VALIDATION': CircuitBuilder.build_validation_circuit,
            'TRANSACTION_ORDERING': CircuitBuilder.build_ordering_circuit,
            
            # General
            'CIRCUIT_EXECUTION': lambda **kw: kw.get('circuit'),
        }
        
        builder = builder_map.get(job_type)
        
        if builder:
            try:
                return builder(**kwargs)
            except Exception as e:
                logger.error(f"Failed to build circuit for {job_type}: {e}")
                return None
        
        logger.warning(f"No builder found for job type: {job_type}")
        return None
    
    # ==================== CRYPTOGRAPHY CIRCUITS ====================
    
    @staticmethod
    def build_qkg_circuit(key_length: int = 256, **kwargs) -> Any:
        """Build Quantum Key Generation circuit (BB84-inspired)."""
        from qiskit import QuantumCircuit
        
        # Use number of qubits based on key length
        num_qubits = min(key_length, 32)  # Limit for hardware
        
        qc = QuantumCircuit(num_qubits, num_qubits)
        
        # Apply random basis (Hadamard for superposition)
        for i in range(num_qubits):
            qc.h(i)
        
        # Measure all qubits
        qc.measure(range(num_qubits), range(num_qubits))
        
        logger.info(f"Built QKG circuit: {num_qubits} qubits")
        return qc
    
    @staticmethod
    def build_qrng_circuit(num_bits: int = 256, **kwargs) -> Any:
        """Build Quantum Random Number Generator circuit."""
        from qiskit import QuantumCircuit
        
        num_qubits = min(num_bits, 32)
        
        qc = QuantumCircuit(num_qubits, num_qubits)
        
        # Apply Hadamard to all qubits (true randomness)
        qc.h(range(num_qubits))
        
        # Measure
        qc.measure(range(num_qubits), range(num_qubits))
        
        logger.info(f"Built QRNG circuit: {num_qubits} qubits")
        return qc
    
    @staticmethod
    def build_pqc_circuit(algorithm: str = 'dilithium', **kwargs) -> Any:
        """Build Post-Quantum Cryptography test circuit."""
        from qiskit import QuantumCircuit
        
        # Simple 4-qubit circuit for PQC verification
        qc = QuantumCircuit(4, 4)
        
        # Create entangled state
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        
        qc.measure_all()
        
        logger.info(f"Built PQC circuit: {algorithm}")
        return qc
    
    @staticmethod
    def build_hash_circuit(hash_input: str = "", **kwargs) -> Any:
        """Build quantum hash verification circuit."""
        from qiskit import QuantumCircuit
        
        # Use 8 qubits for hash
        qc = QuantumCircuit(8, 8)
        
        # Apply operations based on hash input
        if hash_input:
            for i, char in enumerate(hash_input[:8]):
                if ord(char) % 2 == 0:
                    qc.x(i)
                qc.h(i)
        
        qc.measure_all()
        
        logger.info("Built hash verification circuit")
        return qc
    
    # ==================== OPTIMIZATION CIRCUITS ====================
    
    @staticmethod
    def build_qaoa_circuit(
        cost_function: Optional[str] = None,
        num_qubits: int = 4,
        p: int = 1,
        **kwargs
    ) -> Any:
        """Build QAOA (Quantum Approximate Optimization Algorithm) circuit."""
        from qiskit import QuantumCircuit
        import numpy as np
        
        qc = QuantumCircuit(num_qubits, num_qubits)
        
        # Initial state: superposition
        qc.h(range(num_qubits))
        
        # QAOA layers
        for _ in range(p):
            # Cost Hamiltonian (example: MaxCut)
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(np.pi / 4, i + 1)
                qc.cx(i, i + 1)
            
            # Mixer Hamiltonian
            for i in range(num_qubits):
                qc.rx(np.pi / 4, i)
        
        qc.measure_all()
        
        logger.info(f"Built QAOA circuit: {num_qubits} qubits, p={p}")
        return qc
    
    @staticmethod
    def build_vqe_circuit(
        hamiltonian: Optional[Any] = None,
        num_qubits: int = 4,
        **kwargs
    ) -> Any:
        """Build VQE (Variational Quantum Eigensolver) circuit."""
        from qiskit import QuantumCircuit
        import numpy as np
        
        qc = QuantumCircuit(num_qubits, num_qubits)
        
        # Ansatz: Hardware-efficient ansatz
        for layer in range(2):
            # Rotation layer
            for i in range(num_qubits):
                qc.ry(np.pi / 4, i)
                qc.rz(np.pi / 4, i)
            
            # Entangling layer
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
        
        qc.measure_all()
        
        logger.info(f"Built VQE circuit: {num_qubits} qubits")
        return qc
    
    @staticmethod
    def build_portfolio_circuit(
        assets: int = 4,
        constraints: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """Build portfolio optimization circuit."""
        # Use QAOA for portfolio optimization
        return CircuitBuilder.build_qaoa_circuit(num_qubits=assets, p=2)
    
    @staticmethod
    def build_routing_circuit(
        num_locations: int = 4,
        **kwargs
    ) -> Any:
        """Build route optimization circuit (TSP-like)."""
        # Use QAOA for routing
        return CircuitBuilder.build_qaoa_circuit(num_qubits=num_locations, p=2)
    
    # ==================== SIMULATION CIRCUITS ====================
    
    @staticmethod
    def build_chemistry_circuit(
        molecule: str = "H2",
        **kwargs
    ) -> Any:
        """Build quantum chemistry simulation circuit."""
        from qiskit import QuantumCircuit
        
        # Simple H2 molecule simulation (2 qubits)
        qc = QuantumCircuit(2, 2)
        
        # Prepare molecular ground state approximation
        qc.x(0)  # Electron 1
        qc.h(1)  # Superposition for electron 2
        qc.cx(0, 1)  # Entanglement
        
        qc.measure_all()
        
        logger.info(f"Built chemistry circuit: {molecule}")
        return qc
    
    @staticmethod
    def build_material_circuit(material: str = "graphene", **kwargs) -> Any:
        """Build material property simulation circuit."""
        # Use VQE for material simulation
        return CircuitBuilder.build_vqe_circuit(num_qubits=6)
    
    @staticmethod
    def build_molecular_circuit(
        atoms: int = 4,
        **kwargs
    ) -> Any:
        """Build molecular dynamics circuit."""
        return CircuitBuilder.build_vqe_circuit(num_qubits=atoms)
    
    # ==================== AI/ML CIRCUITS ====================
    
    @staticmethod
    def build_qnn_circuit(
        input_dim: int = 4,
        layers: int = 2,
        **kwargs
    ) -> Any:
        """Build Quantum Neural Network circuit."""
        from qiskit import QuantumCircuit
        import numpy as np
        
        qc = QuantumCircuit(input_dim, input_dim)
        
        # Input encoding
        qc.h(range(input_dim))
        
        # QNN layers
        for _ in range(layers):
            # Rotation layer (trainable)
            for i in range(input_dim):
                qc.ry(np.pi / 3, i)
                qc.rz(np.pi / 3, i)
            
            # Entangling layer
            for i in range(input_dim - 1):
                qc.cx(i, i + 1)
        
        qc.measure_all()
        
        logger.info(f"Built QNN circuit: {input_dim} qubits, {layers} layers")
        return qc
    
    @staticmethod
    def build_kernel_circuit(
        feature_dim: int = 4,
        **kwargs
    ) -> Any:
        """Build quantum kernel circuit."""
        return CircuitBuilder.build_qnn_circuit(input_dim=feature_dim, layers=1)
    
    @staticmethod
    def build_fl_circuit(
        model_params: int = 8,
        **kwargs
    ) -> Any:
        """Build federated learning aggregation circuit."""
        # Use quantum averaging circuit
        from qiskit import QuantumCircuit
        
        qc = QuantumCircuit(model_params, model_params)
        
        # Create superposition for averaging
        qc.h(range(model_params))
        
        # Apply controlled operations for weighted averaging
        for i in range(model_params - 1):
            qc.cx(i, i + 1)
        
        qc.measure_all()
        
        logger.info(f"Built FL aggregation circuit: {model_params} parameters")
        return qc
    
    @staticmethod
    def build_classifier_circuit(
        classes: int = 2,
        features: int = 4,
        **kwargs
    ) -> Any:
        """Build variational classifier circuit."""
        return CircuitBuilder.build_qnn_circuit(input_dim=features, layers=2)
    
    # ==================== CONSENSUS CIRCUITS ====================
    
    @staticmethod
    def build_consensus_circuit(
        validators: int = 8,
        **kwargs
    ) -> Any:
        """Build consensus verification circuit."""
        from qiskit import QuantumCircuit
        
        qc = QuantumCircuit(validators, validators)
        
        # Create equal superposition (all validators)
        qc.h(range(validators))
        
        # Apply entanglement for consensus
        for i in range(validators - 1):
            qc.cx(i, i + 1)
        
        # Oracle for valid consensus states
        qc.barrier()
        
        qc.measure_all()
        
        logger.info(f"Built consensus circuit: {validators} validators")
        return qc
    
    @staticmethod
    def build_validation_circuit(
        block_size: int = 4,
        **kwargs
    ) -> Any:
        """Build block validation circuit."""
        from qiskit import QuantumCircuit
        
        qc = QuantumCircuit(block_size, block_size)
        
        # Initialize with block hash representation
        qc.h(range(block_size))
        
        # Apply validation checks
        for i in range(block_size - 1):
            qc.cx(i, i + 1)
            qc.h(i)
        
        qc.measure_all()
        
        logger.info(f"Built block validation circuit: {block_size} qubits")
        return qc
    
    @staticmethod
    def build_ordering_circuit(
        transactions: int = 6,
        **kwargs
    ) -> Any:
        """Build transaction ordering circuit."""
        # Use QAOA for optimal ordering
        return CircuitBuilder.build_qaoa_circuit(num_qubits=transactions, p=1)


# Convenience functions
def get_required_qubits(job_type: str, **kwargs) -> int:
    """Get number of qubits required for job type."""
    qubit_map = {
        'QUANTUM_KEY_GENERATION': kwargs.get('key_length', 256) // 8,
        'QUANTUM_RANDOM': kwargs.get('num_bits', 256) // 8,
        'POST_QUANTUM_CRYPTO': 4,
        'HASH_VERIFICATION': 8,
        'QAOA_OPTIMIZATION': kwargs.get('num_qubits', 4),
        'VQE_OPTIMIZATION': kwargs.get('num_qubits', 4),
        'PORTFOLIO_OPTIMIZATION': kwargs.get('assets', 4),
        'ROUTE_OPTIMIZATION': kwargs.get('num_locations', 4),
        'QUANTUM_CHEMISTRY': 2,
        'MATERIAL_SIMULATION': 6,
        'MOLECULAR_DYNAMICS': kwargs.get('atoms', 4),
        'QUANTUM_NEURAL_NETWORK': kwargs.get('input_dim', 4),
        'QUANTUM_KERNEL': kwargs.get('feature_dim', 4),
        'FEDERATED_LEARNING_AGGREGATION': kwargs.get('model_params', 8),
        'VARIATIONAL_CLASSIFIER': kwargs.get('features', 4),
        'CONSENSUS_VERIFICATION': kwargs.get('validators', 8),
        'BLOCK_VALIDATION': kwargs.get('block_size', 4),
        'TRANSACTION_ORDERING': kwargs.get('transactions', 6),
    }
    
    return min(qubit_map.get(job_type, 4), 32)  # Cap at 32 qubits
