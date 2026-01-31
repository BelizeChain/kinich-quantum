"""
Quantum Circuit Analyzer for Kinich.

Parses Qiskit QuantumCircuit objects and OpenQASM strings to extract:
- Number of qubits
- Circuit depth
- Gate counts (by type)
- Circuit complexity metrics
- Circuit classification

Supports both Qiskit QuantumCircuit objects and OpenQASM 2.0/3.0 strings.
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging

try:
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logging.warning("Qiskit not installed. Circuit analysis will be limited.")

logger = logging.getLogger(__name__)


class CircuitType(Enum):
    """Type of quantum circuit based on analysis."""
    OPTIMIZATION = "optimization"          # VQE, QAOA
    SIMULATION = "simulation"              # Hamiltonian simulation
    SAMPLING = "sampling"                  # Random sampling, Boson sampling
    SEARCH = "search"                      # Grover's algorithm
    FACTORIZATION = "factorization"        # Shor's algorithm
    ERROR_CORRECTION = "error_correction"  # Surface codes, etc.
    UNKNOWN = "unknown"


@dataclass
class GateStatistics:
    """Statistics about gates in a circuit."""
    total_gates: int
    single_qubit_gates: int
    two_qubit_gates: int
    multi_qubit_gates: int
    gate_counts: Dict[str, int]  # gate_name -> count
    
    @property
    def avg_gates_per_qubit(self) -> float:
        """Average number of gates per qubit."""
        # Computed at call site using num_qubits; keep a safe default
        return float(self.total_gates)


@dataclass
class CircuitMetrics:
    """Complete metrics for a quantum circuit."""
    num_qubits: int
    num_classical_bits: int
    circuit_depth: int
    gate_stats: GateStatistics
    circuit_type: CircuitType
    circuit_hash: str
    complexity_score: float  # 0-100, higher = more complex
    parallelism: float       # 0-1, ratio of parallel gates
    entanglement_depth: int  # Number of two-qubit gate layers
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "qubits": self.num_qubits,
            "classical_bits": self.num_classical_bits,
            "depth": self.circuit_depth,
            "gates": self.gate_stats.total_gates,
            "gate_breakdown": self.gate_stats.gate_counts,
            "single_qubit_gates": self.gate_stats.single_qubit_gates,
            "two_qubit_gates": self.gate_stats.two_qubit_gates,
            "circuit_type": self.circuit_type.value,
            "complexity": round(self.complexity_score, 2),
            "parallelism": round(self.parallelism, 3),
            "entanglement_depth": self.entanglement_depth,
        }


class CircuitAnalyzer:
    """
    Analyze quantum circuits to extract metrics and metadata.
    
    Supports:
    - Qiskit QuantumCircuit objects
    - OpenQASM 2.0 strings
    - OpenQASM 3.0 strings (partial)
    """
    
    def __init__(self):
        """Initialize circuit analyzer."""
        self.single_qubit_gates = {
            'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg',
            'rx', 'ry', 'rz', 'u', 'u1', 'u2', 'u3', 'p',
            'sx', 'sxdg', 'id', 'i'
        }
        
        self.two_qubit_gates = {
            'cx', 'cy', 'cz', 'ch', 'swap', 'iswap',
            'cp', 'crx', 'cry', 'crz', 'cu', 'cu1', 'cu3',
            'dcx', 'ecr', 'rxx', 'ryy', 'rzz', 'rzx',
            'cnot', 'cphase'
        }
        
        self.three_qubit_gates = {
            'ccx', 'cswap', 'ccz', 'toffoli', 'fredkin'
        }
    
    def analyze_circuit(self, circuit: str) -> CircuitMetrics:
        """
        Analyze a quantum circuit.
        
        Args:
            circuit: OpenQASM string or Qiskit QuantumCircuit
        
        Returns:
            CircuitMetrics with complete analysis
        """
        # Check if it's a Qiskit circuit
        if QISKIT_AVAILABLE and isinstance(circuit, QuantumCircuit):
            return self._analyze_qiskit_circuit(circuit)
        
        # Otherwise, treat as OpenQASM string
        if isinstance(circuit, str):
            return self._analyze_qasm_string(circuit)
        
        raise ValueError(f"Unsupported circuit type: {type(circuit)}")
    
    def _analyze_qiskit_circuit(self, circuit: QuantumCircuit) -> CircuitMetrics:
        """Analyze a Qiskit QuantumCircuit object."""
        import hashlib
        
        # Basic circuit properties
        num_qubits = circuit.num_qubits
        num_clbits = circuit.num_clbits
        circuit_depth = circuit.depth()
        
        # Count gates by type
        gate_counts: Dict[str, int] = {}
        single_qubit_count = 0
        two_qubit_count = 0
        multi_qubit_count = 0
        
        for instruction, qargs, cargs in circuit.data:
            gate_name = instruction.name.lower()
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            
            # Categorize by number of qubits
            num_gate_qubits = len(qargs)
            if num_gate_qubits == 1:
                single_qubit_count += 1
            elif num_gate_qubits == 2:
                two_qubit_count += 1
            else:
                multi_qubit_count += 1
        
        total_gates = sum(gate_counts.values())
        
        # Create gate statistics
        gate_stats = GateStatistics(
            total_gates=total_gates,
            single_qubit_gates=single_qubit_count,
            two_qubit_gates=two_qubit_count,
            multi_qubit_gates=multi_qubit_count,
            gate_counts=gate_counts,
        )
        
        # Calculate circuit hash
        circuit_qasm = circuit.qasm()
        circuit_hash = hashlib.sha256(circuit_qasm.encode()).hexdigest()
        
        # Detect circuit type
        circuit_type = self._detect_circuit_type(gate_counts, circuit)
        
        # Calculate complexity score (0-100)
        complexity_score = self._calculate_complexity(
            num_qubits, circuit_depth, gate_stats
        )
        
        # Calculate parallelism (ratio of gates that can run in parallel)
        parallelism = self._calculate_parallelism(circuit)
        
        # Calculate entanglement depth (number of two-qubit gate layers)
        entanglement_depth = self._calculate_entanglement_depth(circuit)
        
        return CircuitMetrics(
            num_qubits=num_qubits,
            num_classical_bits=num_clbits,
            circuit_depth=circuit_depth,
            gate_stats=gate_stats,
            circuit_type=circuit_type,
            circuit_hash=circuit_hash,
            complexity_score=complexity_score,
            parallelism=parallelism,
            entanglement_depth=entanglement_depth,
        )
    
    def _analyze_qasm_string(self, qasm: str) -> CircuitMetrics:
        """
        Analyze an OpenQASM string.
        
        Parses QASM manually if Qiskit is not available,
        or converts to Qiskit circuit for detailed analysis.
        """
        if QISKIT_AVAILABLE:
            # Use Qiskit to parse QASM
            try:
                circuit = QuantumCircuit.from_qasm_str(qasm)
                return self._analyze_qiskit_circuit(circuit)
            except Exception as e:
                logger.warning(f"Failed to parse QASM with Qiskit: {e}")
                # Fall back to manual parsing
        
        # Manual QASM parsing (simplified)
        return self._parse_qasm_manually(qasm)
    
    def _parse_qasm_manually(self, qasm: str) -> CircuitMetrics:
        """
        Manually parse OpenQASM string.
        
        This is a simplified parser for when Qiskit is not available.
        Supports basic OpenQASM 2.0 syntax.
        """
        import hashlib
        
        lines = qasm.strip().split('\n')
        
        # Extract qubit and classical bit declarations
        num_qubits = 0
        num_clbits = 0
        
        for line in lines:
            line = line.strip()
            
            # qreg declaration
            if line.startswith('qreg'):
                match = re.search(r'qreg\s+\w+\[(\d+)\]', line)
                if match:
                    num_qubits += int(match.group(1))
            
            # creg declaration
            elif line.startswith('creg'):
                match = re.search(r'creg\s+\w+\[(\d+)\]', line)
                if match:
                    num_clbits += int(match.group(1))
        
        # Count gates
        gate_counts: Dict[str, int] = {}
        single_qubit_count = 0
        two_qubit_count = 0
        multi_qubit_count = 0
        
        for line in lines:
            line = line.strip()
            
            # Skip comments, declarations, and empty lines
            if not line or line.startswith('//') or line.startswith('OPENQASM') or \
               line.startswith('include') or line.startswith('qreg') or \
               line.startswith('creg') or line.startswith('barrier'):
                continue
            
            # Extract gate name
            gate_match = re.match(r'(\w+)(?:\([\d.,\s]+\))?\s+', line)
            if gate_match:
                gate_name = gate_match.group(1).lower()
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
                
                # Categorize by gate type
                if gate_name in self.single_qubit_gates:
                    single_qubit_count += 1
                elif gate_name in self.two_qubit_gates:
                    two_qubit_count += 1
                elif gate_name in self.three_qubit_gates:
                    multi_qubit_count += 1
                else:
                    # Count arguments to determine qubit count
                    args = re.findall(r'\w+\[\d+\]', line)
                    if len(args) == 1:
                        single_qubit_count += 1
                    elif len(args) == 2:
                        two_qubit_count += 1
                    else:
                        multi_qubit_count += 1
        
        total_gates = sum(gate_counts.values())
        
        # Estimate circuit depth (simplified: assumes sequential execution)
        # In reality, parallel gates reduce depth
        circuit_depth = total_gates  # Conservative estimate
        
        # Create gate statistics
        gate_stats = GateStatistics(
            total_gates=total_gates,
            single_qubit_gates=single_qubit_count,
            two_qubit_gates=two_qubit_count,
            multi_qubit_gates=multi_qubit_count,
            gate_counts=gate_counts,
        )
        
        # Calculate circuit hash
        circuit_hash = hashlib.sha256(qasm.encode()).hexdigest()
        
        # Detect circuit type (simplified)
        circuit_type = self._detect_circuit_type(gate_counts, None)
        
        # Calculate metrics
        complexity_score = self._calculate_complexity(
            num_qubits, circuit_depth, gate_stats
        )
        
        # Simplified metrics (no DAG analysis)
        parallelism = self._estimate_parallelism_fallback(
            total_gates=total_gates,
            num_qubits=num_qubits,
            two_qubit_gates=two_qubit_count,
            multi_qubit_gates=multi_qubit_count,
        )
        entanglement_depth = max(1, two_qubit_count)  # Conservative estimate
        
        return CircuitMetrics(
            num_qubits=num_qubits,
            num_classical_bits=num_clbits,
            circuit_depth=circuit_depth,
            gate_stats=gate_stats,
            circuit_type=circuit_type,
            circuit_hash=circuit_hash,
            complexity_score=complexity_score,
            parallelism=parallelism,
            entanglement_depth=entanglement_depth,
        )
    
    def _detect_circuit_type(
        self,
        gate_counts: Dict[str, int],
        circuit: Optional[QuantumCircuit]
    ) -> CircuitType:
        """
        Detect the type of circuit based on gate patterns.
        
        Uses heuristics:
        - High parameterized gates (rx, ry, rz) → Optimization
        - Many controlled gates → Search
        - Toffoli gates → Factorization
        - High two-qubit gate ratio → Simulation
        """
        total_gates = sum(gate_counts.values())
        if total_gates == 0:
            return CircuitType.UNKNOWN
        
        # Check for parameterized rotation gates (VQE, QAOA)
        param_gates = sum(
            gate_counts.get(g, 0)
            for g in ['rx', 'ry', 'rz', 'u', 'u1', 'u2', 'u3', 'p']
        )
        if param_gates / total_gates > 0.5:
            return CircuitType.OPTIMIZATION
        
        # Check for Toffoli gates (Shor's algorithm, arithmetic)
        if gate_counts.get('ccx', 0) > 0 or gate_counts.get('toffoli', 0) > 0:
            return CircuitType.FACTORIZATION
        
        # Check for high controlled gate usage (Grover's search)
        controlled_gates = sum(
            gate_counts.get(g, 0)
            for g in ['cx', 'cy', 'cz', 'ch', 'cp', 'crx', 'cry', 'crz']
        )
        if controlled_gates / total_gates > 0.6:
            return CircuitType.SEARCH
        
        # Check for swap gates (quantum simulation)
        if gate_counts.get('swap', 0) > 0 or gate_counts.get('iswap', 0) > 0:
            return CircuitType.SIMULATION
        
        # Default
        return CircuitType.SAMPLING
    
    def _calculate_complexity(
        self,
        num_qubits: int,
        circuit_depth: int,
        gate_stats: GateStatistics
    ) -> float:
        """
        Calculate circuit complexity score (0-100).
        
        Factors:
        - Number of qubits (logarithmic weight)
        - Circuit depth (linear weight)
        - Two-qubit gates (higher weight than single-qubit)
        - Total gates
        """
        # Base complexity from qubits (log scale)
        import math
        qubit_complexity = min(100, 20 * math.log2(max(1, num_qubits)))
        
        # Depth complexity (linear, normalized to 100)
        depth_complexity = min(100, circuit_depth * 2)
        
        # Gate complexity (two-qubit gates count more)
        gate_complexity = min(100, (
            gate_stats.single_qubit_gates * 0.5 +
            gate_stats.two_qubit_gates * 2.0 +
            gate_stats.multi_qubit_gates * 5.0
        ))
        
        # Weighted average
        complexity = (
            qubit_complexity * 0.3 +
            depth_complexity * 0.4 +
            gate_complexity * 0.3
        )
        
        return min(100, complexity)
    
    def _calculate_parallelism(self, circuit: QuantumCircuit) -> float:
        """
        Calculate circuit parallelism.
        
        Ratio of gates that can be executed in parallel.
        Uses DAG analysis to find critical path.
        """
        if not QISKIT_AVAILABLE:
            return 0.5  # Default estimate
        
        try:
            from qiskit.converters import circuit_to_dag
            
            dag = circuit_to_dag(circuit)
            
            # Calculate parallelism from DAG
            total_gates = len(circuit.data)
            dag_depth = dag.depth()
            
            if dag_depth == 0 or total_gates == 0:
                return 0.0
            
            # Parallelism = (total gates / depth) / num_qubits
            # Normalized to 0-1 range
            parallelism = (total_gates / dag_depth) / max(1, circuit.num_qubits)
            
            return min(1.0, parallelism)
            
        except Exception as e:
            logger.warning(f"Failed to calculate parallelism: {e}")
            return 0.5

    def _estimate_parallelism_fallback(
        self,
        total_gates: int,
        num_qubits: int,
        two_qubit_gates: int,
        multi_qubit_gates: int,
    ) -> float:
        """Heuristic parallelism when Qiskit DAG is unavailable."""
        if total_gates == 0 or num_qubits == 0:
            return 0.0
        # Estimate layers created by entangling gates; keep at least one layer.
        entangling_layers = max(1, two_qubit_gates + multi_qubit_gates)
        ideal_parallel = total_gates / max(1, num_qubits)
        parallelism = ideal_parallel / entangling_layers
        return min(1.0, max(0.1, parallelism))
    
    def _calculate_entanglement_depth(self, circuit: QuantumCircuit) -> int:
        """
        Calculate entanglement depth.
        
        Number of layers of two-qubit gates that create entanglement.
        """
        if not QISKIT_AVAILABLE:
            return 1  # Default estimate
        
        try:
            from qiskit.converters import circuit_to_dag
            
            dag = circuit_to_dag(circuit)
            
            # Count layers with two-qubit gates
            entanglement_layers = 0
            
            for layer in dag.layers():
                has_two_qubit_gate = False
                for node in layer['graph'].op_nodes():
                    if len(node.qargs) >= 2:
                        has_two_qubit_gate = True
                        break
                
                if has_two_qubit_gate:
                    entanglement_layers += 1
            
            return entanglement_layers
            
        except Exception as e:
            logger.warning(f"Failed to calculate entanglement depth: {e}")
            return 1
    
    def validate_circuit(self, circuit: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a circuit for syntax and structural errors.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Try to analyze the circuit
            metrics = self.analyze_circuit(circuit)
            
            # Basic validation checks
            if metrics.num_qubits == 0:
                return False, "Circuit has no qubits"
            
            if metrics.gate_stats.total_gates == 0:
                return False, "Circuit has no gates"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
