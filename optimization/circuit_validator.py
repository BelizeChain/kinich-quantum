"""
Circuit validation before quantum execution
Enforces hardware limits and optimizes circuits
"""

import re
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CircuitValidator:
    """
    Validate quantum circuits before execution
    Ensures circuits meet hardware constraints
    """
    
    # Hardware limits (based on current Azure/IBM quantum processors)
    MAX_QUBITS = 20
    MAX_CIRCUIT_DEPTH = 100
    MAX_GATES = 1000
    
    def __init__(self, backend: str = "azure"):
        self.backend = backend
        
        # Backend-specific limits
        if backend == "azure":
            self.MAX_QUBITS = 20  # Azure Quantum IonQ max
            self.MAX_CIRCUIT_DEPTH = 100
        elif backend == "ibm":
            self.MAX_QUBITS = 27  # IBM Quantum Eagle processors
            self.MAX_CIRCUIT_DEPTH = 200
        elif backend == "simulator":
            self.MAX_QUBITS = 30  # Simulators can handle more
            self.MAX_CIRCUIT_DEPTH = 500
    
    def validate(self, circuit_qasm: str) -> Tuple[bool, Optional[str]]:
        """
        Validate QASM circuit
        
        Returns:
            (is_valid, error_message)
        """
        # Parse QASM
        try:
            num_qubits = self._count_qubits(circuit_qasm)
            circuit_depth = self._estimate_depth(circuit_qasm)
            num_gates = self._count_gates(circuit_qasm)
        except Exception as e:
            return False, f"Invalid QASM syntax: {e}"
        
        # Check qubit limit
        if num_qubits > self.MAX_QUBITS:
            return False, f"Circuit uses {num_qubits} qubits (max: {self.MAX_QUBITS})"
        
        # Check depth limit
        if circuit_depth > self.MAX_CIRCUIT_DEPTH:
            return False, f"Circuit depth {circuit_depth} exceeds limit ({self.MAX_CIRCUIT_DEPTH})"
        
        # Check gate count
        if num_gates > self.MAX_GATES:
            return False, f"Circuit has {num_gates} gates (max: {self.MAX_GATES})"
        
        logger.info(f"✅ Circuit validated: {num_qubits} qubits, depth {circuit_depth}, {num_gates} gates")
        return True, None
    
    def _count_qubits(self, qasm: str) -> int:
        """Count number of qubits in QASM circuit"""
        # Look for qreg declarations
        match = re.search(r'qreg\s+\w+\[(\d+)\]', qasm)
        if match:
            return int(match.group(1))
        return 0
    
    def _estimate_depth(self, qasm: str) -> int:
        """Estimate circuit depth from QASM"""
        # Simplified depth estimation
        # In production, use proper QASM parser
        gates = re.findall(r'^(h|x|y|z|cx|cz|ccx|measure)\s+', qasm, re.MULTILINE)
        return len(gates)  # Rough approximation
    
    def _count_gates(self, qasm: str) -> int:
        """Count total gates in circuit"""
        gates = re.findall(r'^(h|x|y|z|cx|cz|ccx|rx|ry|rz|swap)\s+', qasm, re.MULTILINE)
        return len(gates)
    
    def suggest_optimizations(self, circuit_qasm: str) -> list[str]:
        """Suggest circuit optimizations"""
        suggestions = []
        
        depth = self._estimate_depth(circuit_qasm)
        if depth > 50:
            suggestions.append("High circuit depth - consider circuit compression")
        
        num_gates = self._count_gates(circuit_qasm)
        if num_gates > 500:
            suggestions.append("Many gates - consider gate fusion optimization")
        
        # Check for redundant gates (simplified)
        if "h q[0]; h q[0];" in circuit_qasm:
            suggestions.append("Redundant Hadamard gates detected - can be eliminated")
        
        return suggestions


class CircuitOptimizer:
    """
    Optimize quantum circuits for hardware execution
    Reduces gate count and circuit depth
    """
    
    def optimize(self, circuit_qasm: str) -> str:
        """
        Apply basic circuit optimizations
        
        Args:
            circuit_qasm: Input QASM circuit
        
        Returns:
            Optimized QASM circuit
        """
        optimized = circuit_qasm
        
        # Remove consecutive inverse gates (e.g., H H = I)
        optimized = re.sub(r'h q\[(\d+)\];\s*h q\[\1\];', '', optimized)
        optimized = re.sub(r'x q\[(\d+)\];\s*x q\[\1\];', '', optimized)
        
        # Merge consecutive single-qubit rotations
        # (Simplified - in production use Qiskit transpiler)
        
        logger.info("Circuit optimized")
        return optimized


# Example usage
if __name__ == "__main__":
    qasm_circuit = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    
    h q[0];
    cx q[0],q[1];
    cx q[1],q[2];
    
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    """
    
    validator = CircuitValidator(backend="azure")
    is_valid, error = validator.validate(qasm_circuit)
    
    if is_valid:
        print("✅ Circuit is valid")
        suggestions = validator.suggest_optimizations(qasm_circuit)
        if suggestions:
            print("Optimization suggestions:")
            for s in suggestions:
                print(f"  - {s}")
    else:
        print(f"❌ Invalid circuit: {error}")
