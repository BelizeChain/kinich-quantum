"""
Test Suite for Kinich Circuit Analysis and Cost Calculation

Tests the CircuitAnalyzer and CostCalculator components that replaced
hardcoded circuit metadata in the Kinich API.

Author: BelizeChain Team
Date: October 2025
"""

import pytest
from kinich.circuit_analyzer import (
    CircuitAnalyzer,
    CircuitMetrics,
    CircuitType,
    GateStatistics,
)
from kinich.cost_calculator import (
    CostCalculator,
    BackendProvider,
)


class TestCircuitAnalyzer:
    """Test CircuitAnalyzer functionality."""
    
    def setup_method(self):
        """Initialize analyzer before each test."""
        self.analyzer = CircuitAnalyzer()
    
    def test_analyze_simple_bell_state(self):
        """Test analyzing a simple Bell state circuit (OpenQASM)."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        assert metrics.num_qubits == 2, f"Expected 2 qubits, got {metrics.num_qubits}"
        assert metrics.circuit_depth >= 2, f"Expected depth >= 2, got {metrics.circuit_depth}"
        assert metrics.gate_stats.total_gates >= 2, f"Expected >= 2 gates, got {metrics.gate_stats.total_gates}"
        assert 'h' in metrics.gate_stats.gate_counts, "Expected Hadamard gate"
        assert 'cx' in metrics.gate_stats.gate_counts, "Expected CNOT gate"
        assert metrics.gate_stats.single_qubit_gates >= 1, "Expected single-qubit gates"
        assert metrics.gate_stats.two_qubit_gates >= 1, "Expected two-qubit gates"
    
    def test_analyze_ghz_state(self):
        """Test analyzing a 3-qubit GHZ state circuit."""
        qasm = """
        OPENQASM 2.0;
        qreg q[3];
        h q[0];
        cx q[0],q[1];
        cx q[1],q[2];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        assert metrics.num_qubits == 3, f"Expected 3 qubits, got {metrics.num_qubits}"
        assert metrics.circuit_depth >= 2, f"Expected depth >= 2, got {metrics.circuit_depth}"
        assert metrics.gate_stats.total_gates == 3, f"Expected 3 gates (1 h + 2 cx), got {metrics.gate_stats.total_gates}"
        assert metrics.gate_stats.single_qubit_gates == 1, "Expected 1 Hadamard gate"
        assert metrics.gate_stats.two_qubit_gates == 2, "Expected 2 CNOT gates"
    
    def test_circuit_type_detection(self):
        """Test circuit type detection heuristics."""
        # Optimization circuit (T-gates, controlled gates)
        optimization_qasm = """
        OPENQASM 2.0;
        qreg q[3];
        t q[0];
        t q[1];
        cx q[0],q[1];
        tdg q[2];
        """
        
        metrics = self.analyzer.analyze_circuit(optimization_qasm)
        # Circuit type detection is heuristic-based, so we just verify it's set
        assert metrics.circuit_type is not None, "Circuit type should be detected"
        assert isinstance(metrics.circuit_type, CircuitType), "Should be CircuitType enum"
    
    def test_complexity_calculation(self):
        """Test complexity score calculation (0-100 scale)."""
        # Simple circuit
        simple_qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        cx q[0],q[1];
        """
        
        simple_metrics = self.analyzer.analyze_circuit(simple_qasm)
        
        # Complex circuit
        complex_qasm = """
        OPENQASM 2.0;
        qreg q[10];
        h q[0];
        cx q[0],q[1];
        cx q[1],q[2];
        cx q[2],q[3];
        cx q[3],q[4];
        t q[5];
        tdg q[6];
        ccx q[7],q[8],q[9];
        """
        
        complex_metrics = self.analyzer.analyze_circuit(complex_qasm)
        
        # Verify complexity is in valid range
        assert 0 <= simple_metrics.complexity_score <= 100, "Complexity must be 0-100"
        assert 0 <= complex_metrics.complexity_score <= 100, "Complexity must be 0-100"
        
        # Complex circuit should have higher complexity
        assert complex_metrics.complexity_score > simple_metrics.complexity_score, \
            f"Complex circuit ({complex_metrics.complexity_score}) should be more complex than simple ({simple_metrics.complexity_score})"
    
    def test_parallelism_calculation(self):
        """Test parallelism score calculation (0-1 scale)."""
        qasm = """
        OPENQASM 2.0;
        qreg q[3];
        h q[0];
        h q[1];
        h q[2];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        # Verify parallelism is valid
        assert 0 <= metrics.parallelism <= 1, f"Parallelism must be 0-1, got {metrics.parallelism}"
        
        # Independent gates should have high parallelism
        assert metrics.parallelism > 0.5, f"Independent Hadamards should be parallelizable, got {metrics.parallelism}"
    
    def test_entanglement_depth(self):
        """Test entanglement depth calculation."""
        # Circuit with 2 layers of entanglement
        qasm = """
        OPENQASM 2.0;
        qreg q[4];
        h q[0];
        cx q[0],q[1];
        cx q[2],q[3];
        cx q[1],q[2];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        # Should have at least 2 layers of two-qubit gates
        assert metrics.entanglement_depth >= 2, \
            f"Expected entanglement depth >= 2, got {metrics.entanglement_depth}"
    
    def test_invalid_circuit(self):
        """Test handling of invalid circuits."""
        invalid_qasm = "not a valid circuit"
        
        is_valid, error = self.analyzer.validate_circuit(invalid_qasm)
        
        assert not is_valid, "Invalid circuit should fail validation"
        assert error is not None, "Should provide error message"
        assert len(error) > 0, "Error message should not be empty"


class TestCostCalculator:
    """Test CostCalculator functionality."""
    
    def setup_method(self):
        """Initialize calculator before each test."""
        self.calculator = CostCalculator()
        self.analyzer = CircuitAnalyzer()
    
    def test_ionq_cost_calculation(self):
        """Test IonQ backend cost calculation."""
        # Create simple circuit
        qasm = """
        OPENQASM 2.0;
        qreg q[3];
        h q[0];
        cx q[0],q[1];
        cx q[1],q[2];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        cost_usd, cost_dalla = self.calculator.calculate_job_cost(
            BackendProvider.AZURE_IONQ.value,
            metrics,
            num_shots=1000
        )
        
        # Verify cost is reasonable
        assert cost_usd > 0, "Cost should be positive"
        assert cost_dalla > 0, "DALLA cost should be positive"
        
        # Verify USD to DALLA conversion (1 USD = 10,000,000 DALLA)
        expected_dalla = int(cost_usd * 10_000_000)
        assert abs(cost_dalla - expected_dalla) < 100, \
            f"DALLA conversion mismatch: {cost_dalla} vs {expected_dalla}"
    
    def test_quantinuum_minimum_cost(self):
        """Test Quantinuum's $0.50 minimum cost."""
        # Very simple circuit
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        cost_usd, cost_dalla = self.calculator.calculate_job_cost(
            BackendProvider.AZURE_QUANTINUUM.value,
            metrics,
            num_shots=10  # Very few shots
        )
        
        # Should enforce $0.50 minimum
        assert cost_usd >= 0.50, \
            f"Quantinuum should enforce $0.50 minimum, got ${cost_usd}"
    
    def test_simulator_free_cost(self):
        """Test that simulators are free."""
        qasm = """
        OPENQASM 2.0;
        qreg q[5];
        h q[0];
        cx q[0],q[1];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        cost_usd, cost_dalla = self.calculator.calculate_job_cost(
            BackendProvider.QISKIT_SIMULATOR.value,
            metrics,
            num_shots=10000
        )
        
        assert cost_usd == 0.0, f"Simulator should be free, got ${cost_usd}"
        assert cost_dalla == 0, f"Simulator should be free, got {cost_dalla} DALLA"
    
    def test_execution_time_estimation(self):
        """Test execution time estimation."""
        qasm = """
        OPENQASM 2.0;
        qreg q[3];
        h q[0];
        cx q[0],q[1];
        cx q[1],q[2];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        circuit_ms, queue_s, total_s = self.calculator.estimate_execution_time(
            BackendProvider.AZURE_IONQ.value,
            metrics,
            num_shots=1000
        )
        
        # Verify times are positive
        assert circuit_ms > 0, "Circuit time should be positive"
        assert queue_s > 0, "Queue time should be positive"
        assert total_s > 0, "Total time should be positive"
        
        # Total should be sum of circuit + queue
        expected_total = (circuit_ms / 1000) + queue_s
        assert abs(total_s - expected_total) < 0.01, \
            f"Total time mismatch: {total_s} vs {expected_total}"
    
    def test_backend_compatibility_qubit_limit(self):
        """Test backend compatibility checking (qubit limits)."""
        # Circuit with 15 qubits (exceeds IonQ's 11-qubit limit)
        qasm = """
        OPENQASM 2.0;
        qreg q[15];
        h q[0];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        is_compatible, error = self.calculator.validate_backend_compatibility(
            BackendProvider.AZURE_IONQ.value,
            metrics
        )
        
        assert not is_compatible, "15 qubits should exceed IonQ's 11-qubit limit"
        assert error is not None, "Should provide error message"
        assert "qubit" in error.lower(), "Error should mention qubit limit"
    
    def test_backend_recommendation_cost(self):
        """Test backend recommendation (optimize for cost)."""
        qasm = """
        OPENQASM 2.0;
        qreg q[3];
        h q[0];
        cx q[0],q[1];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        recommended = self.calculator.recommend_backend(
            metrics,
            budget_dalla=100_000_000,  # 10 USD = 100M DALLA
            optimize_for="cost"
        )
        
        # Should recommend a backend
        assert recommended is not None, "Should recommend a backend"
        assert isinstance(recommended, str), "Should return backend name string"
        
        # Verify it's a valid backend
        valid_backends = [b.value for b in BackendProvider]
        assert recommended in valid_backends, \
            f"Recommended backend '{recommended}' not in valid backends: {valid_backends}"
    
    def test_backend_recommendation_speed(self):
        """Test backend recommendation (optimize for speed)."""
        qasm = """
        OPENQASM 2.0;
        qreg q[5];
        h q[0];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        recommended = self.calculator.recommend_backend(
            metrics,
            budget_dalla=1_000_000_000,  # 100 USD = 1B DALLA
            optimize_for="speed"
        )
        
        assert recommended is not None, "Should recommend a backend for speed"
        
        # Speed optimization should avoid simulators if budget allows
        if recommended == BackendProvider.QISKIT_SIMULATOR.value:
            # Only acceptable if circuit exceeds all hardware limits
            assert metrics.num_qubits > 127, "Should use hardware if within qubit limits"
    
    def test_backend_recommendation_quality(self):
        """Test backend recommendation (optimize for quality)."""
        qasm = """
        OPENQASM 2.0;
        qreg q[3];
        h q[0];
        cx q[0],q[1];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        recommended = self.calculator.recommend_backend(
            metrics,
            budget_dalla=500_000_000,  # 50 USD = 500M DALLA
            optimize_for="quality"
        )
        
        assert recommended is not None, "Should recommend a backend for quality"
        
        # Quality optimization should prefer high-fidelity backends
        # Quantinuum has highest fidelity (0.995) but expensive
        # With $50 budget, should be able to afford it
        high_quality_backends = [
            BackendProvider.AZURE_QUANTINUUM.value,
            BackendProvider.IBM_QUANTUM.value,
            BackendProvider.AZURE_IONQ.value,
        ]
        assert recommended in high_quality_backends, \
            f"Quality optimization should recommend high-fidelity backend, got {recommended}"
    
    def test_pricing_info_retrieval(self):
        """Test retrieval of pricing information."""
        info = self.calculator.get_pricing_info(BackendProvider.AZURE_IONQ.value)
        
        assert info is not None, "Should return pricing info"
        assert "cost_per_shot" in info, "Should include cost per shot"
        assert "cost_per_qubit" in info, "Should include cost per qubit"
        assert "max_qubits" in info, "Should include max qubits"
        assert info["cost_per_shot"] > 0, "Cost per shot should be positive"
        assert info["max_qubits"] == 11, f"IonQ max qubits should be 11, got {info['max_qubits']}"


class TestIntegration:
    """Integration tests combining analyzer and calculator."""
    
    def setup_method(self):
        """Initialize components."""
        self.analyzer = CircuitAnalyzer()
        self.calculator = CostCalculator()
    
    def test_full_workflow_simple_circuit(self):
        """Test complete workflow: parse → validate → cost → time."""
        qasm = """
        OPENQASM 2.0;
        qreg q[3];
        h q[0];
        cx q[0],q[1];
        cx q[1],q[2];
        """
        
        # 1. Analyze circuit
        metrics = self.analyzer.analyze_circuit(qasm)
        assert metrics is not None, "Circuit analysis should succeed"
        
        # 2. Validate compatibility
        is_compatible, error = self.calculator.validate_backend_compatibility(
            BackendProvider.AZURE_IONQ.value,
            metrics
        )
        assert is_compatible, f"Circuit should be compatible with IonQ: {error}"
        
        # 3. Calculate cost
        cost_usd, cost_dalla = self.calculator.calculate_job_cost(
            BackendProvider.AZURE_IONQ.value,
            metrics,
            num_shots=1000
        )
        assert cost_usd > 0, "Cost should be calculated"
        
        # 4. Estimate execution time
        circuit_ms, queue_s, total_s = self.calculator.estimate_execution_time(
            BackendProvider.AZURE_IONQ.value,
            metrics,
            num_shots=1000
        )
        assert total_s > 0, "Execution time should be estimated"
        
        print(f"\n✅ Full workflow test passed:")
        print(f"   Circuit: {metrics.num_qubits} qubits, {metrics.gate_stats.total_gates} gates, depth {metrics.circuit_depth}")
        print(f"   Cost: ${cost_usd:.6f} USD = {cost_dalla:,} DALLA")
        print(f"   Time: {circuit_ms:.2f}ms circuit + {queue_s:.1f}s queue = {total_s:.1f}s total")
    
    def test_full_workflow_complex_circuit(self):
        """Test workflow with a more complex circuit."""
        qasm = """
        OPENQASM 2.0;
        qreg q[5];
        h q[0];
        h q[1];
        cx q[0],q[2];
        cx q[1],q[3];
        t q[4];
        tdg q[2];
        cx q[3],q[4];
        """
        
        metrics = self.analyzer.analyze_circuit(qasm)
        
        # Try multiple backends
        backends_to_test = [
            BackendProvider.AZURE_IONQ.value,
            BackendProvider.AZURE_RIGETTI.value,
            BackendProvider.IBM_QUANTUM.value,
        ]
        
        print(f"\n✅ Complex circuit analysis:")
        print(f"   {metrics.num_qubits} qubits, {metrics.gate_stats.total_gates} gates, depth {metrics.circuit_depth}")
        print(f"   Complexity: {metrics.complexity_score:.1f}/100")
        print(f"   Parallelism: {metrics.parallelism:.2f}")
        
        for backend in backends_to_test:
            is_compatible, error = self.calculator.validate_backend_compatibility(backend, metrics)
            
            if is_compatible:
                cost_usd, cost_dalla = self.calculator.calculate_job_cost(backend, metrics, num_shots=1000)
                circuit_ms, queue_s, total_s = self.calculator.estimate_execution_time(backend, metrics, num_shots=1000)
                
                print(f"\n   {backend}:")
                print(f"      Cost: ${cost_usd:.6f} USD = {cost_dalla:,} DALLA")
                print(f"      Time: {total_s:.1f}s ({circuit_ms:.2f}ms circuit + {queue_s:.1f}s queue)")
            else:
                print(f"\n   {backend}: ❌ Incompatible - {error}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
