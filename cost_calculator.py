"""
Cost Calculator for Kinich Quantum Jobs.

Calculates real costs for quantum circuit execution on various backends:
- Azure Quantum (IonQ, Quantinuum, Rigetti)
- IBM Quantum
- Local simulators

Also estimates execution times based on circuit complexity and backend capabilities.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

from .circuit_analyzer import CircuitMetrics

logger = logging.getLogger(__name__)


class BackendProvider(Enum):
    """Quantum backend providers."""
    AZURE_IONQ = "azure-ionq"
    AZURE_QUANTINUUM = "azure-quantinuum"
    AZURE_RIGETTI = "azure-rigetti"
    IBM_QUANTUM = "ibm-quantum"
    QISKIT_SIMULATOR = "qiskit"
    LOCAL_SIMULATOR = "local"


@dataclass
class BackendPricing:
    """Pricing model for a quantum backend."""
    provider: BackendProvider
    cost_per_shot: float          # USD per shot
    cost_per_qubit: float          # USD per qubit
    cost_per_gate: float           # USD per gate
    cost_per_circuit: float        # USD flat fee per circuit
    min_cost: float                # Minimum charge
    max_qubits: int               # Maximum qubits supported
    supports_pulse: bool          # Pulse-level control
    
    def calculate_cost(
        self,
        num_qubits: int,
        num_shots: int,
        num_gates: int,
        num_circuits: int = 1
    ) -> float:
        """Calculate total cost for a job."""
        cost = (
            self.cost_per_shot * num_shots +
            self.cost_per_qubit * num_qubits +
            self.cost_per_gate * num_gates +
            self.cost_per_circuit * num_circuits
        )
        
        return max(self.min_cost, cost)


@dataclass
class BackendCapabilities:
    """Capabilities and performance metrics for a backend."""
    provider: BackendProvider
    max_qubits: int
    max_circuit_depth: int
    avg_gate_time_us: float       # Microseconds per gate
    avg_readout_time_ms: float    # Milliseconds per measurement
    avg_queue_time_s: float       # Seconds average queue wait
    fidelity: float               # Gate fidelity (0-1)
    connectivity: str             # "full", "linear", "2d-grid", etc.


class CostCalculator:
    """
    Calculate costs and execution times for quantum circuits.
    
    Uses real pricing models from Azure Quantum and IBM Quantum
    (as of January 2025).
    """
    
    def __init__(self):
        """Initialize cost calculator with pricing data."""
        # Azure Quantum pricing (USD, as of 2025)
        self.pricing = {
            BackendProvider.AZURE_IONQ: BackendPricing(
                provider=BackendProvider.AZURE_IONQ,
                cost_per_shot=0.00003,      # $0.03 per 1000 shots
                cost_per_qubit=0.00001,     # $0.01 per qubit
                cost_per_gate=0.0,          # Included in shot cost
                cost_per_circuit=0.0,       # No circuit fee
                min_cost=0.01,              # $0.01 minimum
                max_qubits=11,              # IonQ Aria
                supports_pulse=False,
            ),
            BackendProvider.AZURE_QUANTINUUM: BackendPricing(
                provider=BackendProvider.AZURE_QUANTINUUM,
                cost_per_shot=0.0001,       # $0.10 per 1000 shots
                cost_per_qubit=0.00005,     # $0.05 per qubit
                cost_per_gate=0.0,
                cost_per_circuit=0.1,       # $0.10 per circuit
                min_cost=0.50,              # $0.50 minimum
                max_qubits=20,              # H1-1
                supports_pulse=True,
            ),
            BackendProvider.AZURE_RIGETTI: BackendPricing(
                provider=BackendProvider.AZURE_RIGETTI,
                cost_per_shot=0.00002,      # $0.02 per 1000 shots
                cost_per_qubit=0.000005,    # $0.005 per qubit
                cost_per_gate=0.0,
                cost_per_circuit=0.0,
                min_cost=0.01,
                max_qubits=40,              # Aspen-M-3
                supports_pulse=True,
            ),
            BackendProvider.IBM_QUANTUM: BackendPricing(
                provider=BackendProvider.IBM_QUANTUM,
                cost_per_shot=0.00001,      # $0.01 per 1000 shots (premium)
                cost_per_qubit=0.0,         # Included
                cost_per_gate=0.0,
                cost_per_circuit=0.0,
                min_cost=0.0,               # Free tier available
                max_qubits=127,             # IBM Quantum Eagle
                supports_pulse=True,
            ),
            BackendProvider.QISKIT_SIMULATOR: BackendPricing(
                provider=BackendProvider.QISKIT_SIMULATOR,
                cost_per_shot=0.0,          # Free
                cost_per_qubit=0.0,
                cost_per_gate=0.0,
                cost_per_circuit=0.0,
                min_cost=0.0,
                max_qubits=64,              # Memory limited
                supports_pulse=False,
            ),
            BackendProvider.LOCAL_SIMULATOR: BackendPricing(
                provider=BackendProvider.LOCAL_SIMULATOR,
                cost_per_shot=0.0,
                cost_per_qubit=0.0,
                cost_per_gate=0.0,
                cost_per_circuit=0.0,
                min_cost=0.0,
                max_qubits=32,
                supports_pulse=False,
            ),
        }
        
        # Backend capabilities
        self.capabilities = {
            BackendProvider.AZURE_IONQ: BackendCapabilities(
                provider=BackendProvider.AZURE_IONQ,
                max_qubits=11,
                max_circuit_depth=1000,
                avg_gate_time_us=10.0,
                avg_readout_time_ms=1.0,
                avg_queue_time_s=60.0,
                fidelity=0.99,
                connectivity="full",
            ),
            BackendProvider.AZURE_QUANTINUUM: BackendCapabilities(
                provider=BackendProvider.AZURE_QUANTINUUM,
                max_qubits=20,
                max_circuit_depth=5000,
                avg_gate_time_us=5.0,
                avg_readout_time_ms=0.5,
                avg_queue_time_s=120.0,
                fidelity=0.995,
                connectivity="full",
            ),
            BackendProvider.AZURE_RIGETTI: BackendCapabilities(
                provider=BackendProvider.AZURE_RIGETTI,
                max_qubits=40,
                max_circuit_depth=2000,
                avg_gate_time_us=50.0,
                avg_readout_time_ms=2.0,
                avg_queue_time_s=30.0,
                fidelity=0.98,
                connectivity="2d-grid",
            ),
            BackendProvider.IBM_QUANTUM: BackendCapabilities(
                provider=BackendProvider.IBM_QUANTUM,
                max_qubits=127,
                max_circuit_depth=3000,
                avg_gate_time_us=100.0,
                avg_readout_time_ms=1.5,
                avg_queue_time_s=300.0,
                fidelity=0.999,
                connectivity="heavy-hex",
            ),
            BackendProvider.QISKIT_SIMULATOR: BackendCapabilities(
                provider=BackendProvider.QISKIT_SIMULATOR,
                max_qubits=64,
                max_circuit_depth=10000,
                avg_gate_time_us=0.001,  # Very fast simulation
                avg_readout_time_ms=0.001,
                avg_queue_time_s=0.0,
                fidelity=1.0,  # Perfect simulation
                connectivity="full",
            ),
            BackendProvider.LOCAL_SIMULATOR: BackendCapabilities(
                provider=BackendProvider.LOCAL_SIMULATOR,
                max_qubits=32,
                max_circuit_depth=10000,
                avg_gate_time_us=0.01,
                avg_readout_time_ms=0.01,
                avg_queue_time_s=0.0,
                fidelity=1.0,
                connectivity="full",
            ),
        }
        
        # DALLA token pricing (for BelizeChain billing)
        self.usd_to_dalla = 10_000_000  # 1 USD = 10M DALLA (0.0000001 USD per DALLA)
    
    def calculate_job_cost(
        self,
        backend: str,
        circuit_metrics: CircuitMetrics,
        num_shots: int,
        num_circuits: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate cost for a quantum job.
        
        Args:
            backend: Backend identifier (e.g., "azure-ionq")
            circuit_metrics: Analyzed circuit metrics
            num_shots: Number of measurements
            num_circuits: Number of circuits (for batching)
        
        Returns:
            (cost_usd, cost_dalla)
        """
        # Parse backend provider
        try:
            provider = BackendProvider(backend.lower())
        except ValueError:
            logger.warning(f"Unknown backend: {backend}, using local simulator")
            provider = BackendProvider.LOCAL_SIMULATOR
        
        # Get pricing model
        pricing = self.pricing.get(provider)
        if not pricing:
            logger.warning(f"No pricing for {provider}, assuming free")
            return 0.0, 0.0
        
        # Calculate USD cost
        cost_usd = pricing.calculate_cost(
            num_qubits=circuit_metrics.num_qubits,
            num_shots=num_shots,
            num_gates=circuit_metrics.gate_stats.total_gates,
            num_circuits=num_circuits
        )
        
        # Convert to DALLA
        cost_dalla = int(cost_usd * self.usd_to_dalla)
        
        logger.debug(
            f"Cost for {backend}: ${cost_usd:.6f} USD = {cost_dalla:,} DALLA "
            f"({circuit_metrics.num_qubits} qubits, {num_shots} shots)"
        )
        
        return cost_usd, cost_dalla
    
    def estimate_execution_time(
        self,
        backend: str,
        circuit_metrics: CircuitMetrics,
        num_shots: int
    ) -> Tuple[float, float, float]:
        """
        Estimate execution time for a job.
        
        Returns:
            (circuit_time_ms, queue_time_s, total_time_s)
        """
        # Parse backend provider
        try:
            provider = BackendProvider(backend.lower())
        except ValueError:
            provider = BackendProvider.LOCAL_SIMULATOR
        
        # Get capabilities
        caps = self.capabilities.get(provider)
        if not caps:
            # Default estimates
            return 1000.0, 0.0, 1.0
        
        # Calculate circuit execution time
        gate_time_ms = (
            circuit_metrics.gate_stats.total_gates *
            caps.avg_gate_time_us / 1000.0
        )
        
        readout_time_ms = caps.avg_readout_time_ms * num_shots
        
        circuit_time_ms = gate_time_ms + readout_time_ms
        
        # Queue time estimate
        queue_time_s = caps.avg_queue_time_s
        
        # Total time
        total_time_s = queue_time_s + (circuit_time_ms / 1000.0)
        
        return circuit_time_ms, queue_time_s, total_time_s
    
    def validate_backend_compatibility(
        self,
        backend: str,
        circuit_metrics: CircuitMetrics
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if circuit is compatible with backend.
        
        Returns:
            (is_compatible, error_message)
        """
        try:
            provider = BackendProvider(backend.lower())
        except ValueError:
            return False, f"Unknown backend: {backend}"
        
        pricing = self.pricing.get(provider)
        if not pricing:
            return False, f"No pricing information for {backend}"
        
        # Check qubit limit
        if circuit_metrics.num_qubits > pricing.max_qubits:
            return False, (
                f"Circuit requires {circuit_metrics.num_qubits} qubits, "
                f"but {backend} supports max {pricing.max_qubits}"
            )
        
        # Check circuit depth (soft limit)
        caps = self.capabilities.get(provider)
        if caps and circuit_metrics.circuit_depth > caps.max_circuit_depth:
            logger.warning(
                f"Circuit depth {circuit_metrics.circuit_depth} exceeds "
                f"recommended max {caps.max_circuit_depth} for {backend}"
            )
            # Don't fail, just warn
        
        return True, None
    
    def recommend_backend(
        self,
        circuit_metrics: CircuitMetrics,
        budget_dalla: Optional[int] = None,
        optimize_for: str = "cost"  # "cost", "speed", "quality"
    ) -> str:
        """
        Recommend the best backend for a circuit.
        
        Args:
            circuit_metrics: Analyzed circuit
            budget_dalla: Optional budget constraint
            optimize_for: Optimization criteria
        
        Returns:
            Backend identifier
        """
        compatible_backends = []
        
        # Check all backends for compatibility
        for provider in BackendProvider:
            is_compat, _ = self.validate_backend_compatibility(
                provider.value, circuit_metrics
            )
            if is_compat:
                compatible_backends.append(provider)
        
        if not compatible_backends:
            logger.warning("No compatible backends found, using local simulator")
            return BackendProvider.LOCAL_SIMULATOR.value
        
        # Apply budget constraint
        if budget_dalla:
            budget_usd = budget_dalla / self.usd_to_dalla
            affordable = []
            
            for provider in compatible_backends:
                cost_usd, _ = self.calculate_job_cost(
                    provider.value, circuit_metrics, num_shots=1000
                )
                if cost_usd <= budget_usd:
                    affordable.append(provider)
            
            if affordable:
                compatible_backends = affordable
        
        # Optimize based on criteria
        if optimize_for == "cost":
            # Choose cheapest
            best = min(
                compatible_backends,
                key=lambda p: self.calculate_job_cost(
                    p.value, circuit_metrics, 1000
                )[0]
            )
        
        elif optimize_for == "speed":
            hardware_only = [
                p for p in compatible_backends
                if p not in (BackendProvider.QISKIT_SIMULATOR, BackendProvider.LOCAL_SIMULATOR)
            ]
            candidates = hardware_only if hardware_only else compatible_backends
            # Choose fastest
            best = min(
                candidates,
                key=lambda p: self.estimate_execution_time(
                    p.value, circuit_metrics, 1000
                )[2]
            )
        
        elif optimize_for == "quality":
            # Prefer real hardware when optimizing for fidelity
            hardware_only = [
                p for p in compatible_backends
                if p not in (BackendProvider.QISKIT_SIMULATOR, BackendProvider.LOCAL_SIMULATOR)
            ]
            candidates = hardware_only if hardware_only else compatible_backends
            best = max(
                candidates,
                key=lambda p: self.capabilities.get(
                    p, BackendCapabilities(p, 0, 0, 0, 0, 0, 0.0, "")
                ).fidelity
            )
        
        else:
            # Default: cost-optimized
            best = compatible_backends[0]
        
        logger.info(f"Recommended backend: {best.value} (optimized for {optimize_for})")
        
        return best.value
    
    def get_pricing_info(self, backend: str) -> Dict:
        """Get pricing information for a backend."""
        try:
            provider = BackendProvider(backend.lower())
        except ValueError:
            return {}
        
        pricing = self.pricing.get(provider)
        if not pricing:
            return {}
        
        return {
            "provider": provider.value,
            "cost_per_shot": pricing.cost_per_shot,
            "cost_per_1000_shots": pricing.cost_per_shot * 1000,
            "cost_per_qubit": pricing.cost_per_qubit,
            "cost_per_gate": pricing.cost_per_gate,
            "min_cost_usd": pricing.min_cost,
            "max_qubits": pricing.max_qubits,
            "usd_to_dalla": self.usd_to_dalla,
        }
