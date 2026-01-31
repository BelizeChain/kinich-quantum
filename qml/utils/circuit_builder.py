"""
Quantum Circuit Builder Utilities
"""

import numpy as np
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import Parameter, ParameterVector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class CircuitBuilder:
    """
    Utility for building quantum circuits.
    
    Provides helper methods for common circuit patterns.
    """
    
    @staticmethod
    def create_parameterized_circuit(
        num_qubits: int,
        num_params: int,
        include_measurements: bool = True
    ) -> Tuple:
        """
        Create parameterized quantum circuit.
        
        Args:
            num_qubits: Number of qubits
            num_params: Number of parameters
            include_measurements: Add measurement gates
            
        Returns:
            (circuit, parameters)
        """
        if not QISKIT_AVAILABLE:
            logger.warning("Qiskit not available")
            return None, None
        
        qr = QuantumRegister(num_qubits, 'q')
        circuit = QuantumCircuit(qr)
        
        if include_measurements:
            cr = ClassicalRegister(num_qubits, 'c')
            circuit.add_register(cr)
        
        params = ParameterVector('θ', num_params)
        
        return circuit, params
    
    @staticmethod
    def add_rotation_layer(
        circuit: 'QuantumCircuit',
        parameters: List['Parameter'],
        rotation_gate: str = 'ry'
    ) -> None:
        """
        Add rotation layer to circuit.
        
        Args:
            circuit: Quantum circuit
            parameters: Rotation parameters
            rotation_gate: Gate type ('rx', 'ry', 'rz')
        """
        num_qubits = circuit.num_qubits
        
        for i in range(num_qubits):
            if rotation_gate == 'rx':
                circuit.rx(parameters[i], i)
            elif rotation_gate == 'ry':
                circuit.ry(parameters[i], i)
            elif rotation_gate == 'rz':
                circuit.rz(parameters[i], i)
    
    @staticmethod
    def add_entangling_layer(
        circuit: 'QuantumCircuit',
        entanglement: str = 'linear'
    ) -> None:
        """
        Add entangling layer.
        
        Args:
            circuit: Quantum circuit
            entanglement: Pattern ('linear', 'circular', 'full')
        """
        num_qubits = circuit.num_qubits
        
        if entanglement == 'linear':
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
        
        elif entanglement == 'circular':
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
            circuit.cx(num_qubits - 1, 0)
        
        elif entanglement == 'full':
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    circuit.cx(i, j)
    
    @staticmethod
    def get_circuit_depth(circuit: Optional['QuantumCircuit']) -> int:
        """Get circuit depth."""
        if circuit is None or not QISKIT_AVAILABLE:
            return 0
        return circuit.depth()
