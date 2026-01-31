"""
Adapter Registry for Kinich

Manages multiple quantum backend adapters and provides
unified interface for job execution across different providers.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass
import logging
from enum import Enum

from .qiskit_adapter import QiskitAdapter
from .azure_adapter import AzureAdapter

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Types of quantum backends."""
    
    AZURE_QUANTUM = "azure_quantum"
    IBM_QUANTUM = "ibm_quantum"
    GOOGLE_CIRQ = "google_cirq"
    SPINQ = "spinq"
    QISKIT_AER = "qiskit_aer"
    LOCAL_SIMULATOR = "local_simulator"


@dataclass
class BackendCapabilities:
    """Capabilities of a quantum backend."""
    
    max_qubits: int
    supports_measurement: bool = True
    supports_reset: bool = True
    supports_conditional: bool = False
    coupling_map: Optional[List[List[int]]] = None
    basis_gates: List[str] = None
    is_simulator: bool = True
    cost_per_shot: float = 0.0
    
    def __post_init__(self):
        if self.basis_gates is None:
            self.basis_gates = ['u1', 'u2', 'u3', 'cx']


class AdapterRegistry:
    """
    Registry for managing multiple quantum backend adapters.
    
    Provides unified interface for executing quantum jobs across
    different providers (Azure, IBM, Google, SpinQ, local).
    """
    
    def __init__(self):
        """Initialize adapter registry."""
        self._adapters: Dict[str, Any] = {}
        self._adapter_types: Dict[BackendType, Type] = {
            BackendType.QISKIT_AER: QiskitAdapter,
            BackendType.AZURE_QUANTUM: AzureAdapter,
        }
        self._capabilities: Dict[str, BackendCapabilities] = {}
        self._health_status: Dict[str, bool] = {}
        
        logger.info("Initialized Kinich adapter registry")
    
    def register_adapter(
        self,
        name: str,
        adapter: Any,
        capabilities: Optional[BackendCapabilities] = None
    ) -> None:
        """
        Register a quantum backend adapter.
        
        Args:
            name: Unique name for the adapter
            adapter: Adapter instance
            capabilities: Backend capabilities (auto-detected if None)
        """
        if name in self._adapters:
            logger.warning(f"Adapter '{name}' already registered, replacing")
        
        self._adapters[name] = adapter
        
        # Auto-detect capabilities if not provided
        if capabilities is None:
            capabilities = self._detect_capabilities(adapter)
        
        self._capabilities[name] = capabilities
        self._health_status[name] = True
        
        logger.info(
            f"Registered adapter '{name}': "
            f"{capabilities.max_qubits} qubits, "
            f"{'simulator' if capabilities.is_simulator else 'hardware'}"
        )
    
    def _detect_capabilities(self, adapter: Any) -> BackendCapabilities:
        """Auto-detect adapter capabilities."""
        # Try to get backend info
        try:
            if hasattr(adapter, 'get_backend_info'):
                info = adapter.get_backend_info(adapter.config.default_backend)
                
                return BackendCapabilities(
                    max_qubits=info.get('num_qubits', 32),
                    basis_gates=info.get('basis_gates', ['u1', 'u2', 'u3', 'cx']),
                    coupling_map=info.get('coupling_map'),
                    is_simulator=info.get('is_simulator', True),
                    cost_per_shot=0.0001 if not info.get('is_simulator') else 0.0,
                )
        except Exception as e:
            logger.warning(f"Failed to detect capabilities: {e}")
        
        # Default capabilities
        return BackendCapabilities(max_qubits=32, is_simulator=True)
    
    def unregister_adapter(self, name: str) -> None:
        """Unregister an adapter."""
        if name in self._adapters:
            del self._adapters[name]
            del self._capabilities[name]
            del self._health_status[name]
            logger.info(f"Unregistered adapter '{name}'")
    
    def get_adapter(self, name: str) -> Optional[Any]:
        """Get adapter by name."""
        return self._adapters.get(name)
    
    def list_adapters(self) -> List[str]:
        """List all registered adapter names."""
        return list(self._adapters.keys())
    
    def get_capabilities(self, name: str) -> Optional[BackendCapabilities]:
        """Get capabilities of an adapter."""
        return self._capabilities.get(name)
    
    def find_suitable_adapter(
        self,
        min_qubits: int,
        prefer_simulator: bool = True,
        max_cost: Optional[float] = None
    ) -> Optional[str]:
        """
        Find suitable adapter based on requirements.
        
        Args:
            min_qubits: Minimum required qubits
            prefer_simulator: Prefer simulator over hardware
            max_cost: Maximum acceptable cost per shot
        
        Returns:
            Name of suitable adapter or None
        """
        candidates = []
        
        for name, caps in self._capabilities.items():
            # Check if adapter is healthy
            if not self._health_status.get(name, False):
                continue
            
            # Check qubit requirement
            if caps.max_qubits < min_qubits:
                continue
            
            # Check cost limit
            if max_cost is not None and caps.cost_per_shot > max_cost:
                continue
            
            # Check simulator preference
            score = 0
            if caps.is_simulator == prefer_simulator:
                score += 10
            
            # Prefer more qubits
            score += caps.max_qubits / 100
            
            # Prefer lower cost
            score -= caps.cost_per_shot * 1000
            
            candidates.append((name, score))
        
        if not candidates:
            logger.warning(f"No suitable adapter found for {min_qubits} qubits")
            return None
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        best = candidates[0][0]
        logger.info(f"Selected adapter '{best}' for job requirements")
        return best
    
    def execute_on_best_adapter(
        self,
        circuit: Any,
        min_qubits: Optional[int] = None,
        shots: int = 1024,
        prefer_simulator: bool = True,
        max_cost: Optional[float] = None,
        backend_preference: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute circuit on best available adapter.
        
        Args:
            circuit: Quantum circuit
            min_qubits: Minimum qubits required
            shots: Number of shots
            prefer_simulator: Prefer simulator
            max_cost: Maximum cost per shot
            backend_preference: Preferred backends in order
        
        Returns:
            Execution result
        """
        # Try preferred backends first
        if backend_preference:
            for backend_name in backend_preference:
                if backend_name in self._adapters:
                    adapter = self._adapters[backend_name]
                    
                    try:
                        logger.info(f"Executing on preferred backend: {backend_name}")
                        result = adapter.execute_circuit(circuit, shots=shots)
                        result['adapter_used'] = backend_name
                        return result
                    except Exception as e:
                        logger.warning(f"Execution failed on {backend_name}: {e}")
                        continue
        
        # Auto-select adapter
        if min_qubits is None:
            min_qubits = getattr(circuit, 'num_qubits', 2)
        
        adapter_name = self.find_suitable_adapter(
            min_qubits=min_qubits,
            prefer_simulator=prefer_simulator,
            max_cost=max_cost
        )
        
        if adapter_name is None:
            return {
                'success': False,
                'error': 'No suitable adapter available',
                'min_qubits': min_qubits,
            }
        
        adapter = self._adapters[adapter_name]
        
        try:
            result = adapter.execute_circuit(circuit, shots=shots)
            result['adapter_used'] = adapter_name
            return result
        except Exception as e:
            logger.error(f"Execution failed on {adapter_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'adapter': adapter_name,
            }
    
    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health check on all adapters.
        
        Returns:
            Health status for each adapter
        """
        results = {}
        
        for name, adapter in self._adapters.items():
            try:
                if hasattr(adapter, 'health_check'):
                    health = adapter.health_check()
                    results[name] = health
                    self._health_status[name] = health.get('connected', False)
                else:
                    results[name] = {'status': 'unknown', 'message': 'No health_check method'}
                    self._health_status[name] = True  # Assume healthy
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = {'status': 'error', 'error': str(e)}
                self._health_status[name] = False
        
        return results
    
    def get_registry_info(self) -> Dict[str, Any]:
        """Get information about the registry."""
        return {
            'total_adapters': len(self._adapters),
            'healthy_adapters': sum(1 for h in self._health_status.values() if h),
            'adapters': {
                name: {
                    'healthy': self._health_status.get(name, False),
                    'capabilities': {
                        'max_qubits': self._capabilities[name].max_qubits,
                        'is_simulator': self._capabilities[name].is_simulator,
                        'cost_per_shot': self._capabilities[name].cost_per_shot,
                    }
                }
                for name in self._adapters.keys()
            }
        }


# Global registry instance
_global_registry: Optional[AdapterRegistry] = None


def get_global_registry() -> AdapterRegistry:
    """Get global adapter registry (singleton)."""
    global _global_registry
    
    if _global_registry is None:
        _global_registry = AdapterRegistry()
    
    return _global_registry


def register_default_adapters() -> AdapterRegistry:
    """
    Register default adapters based on available packages.
    
    Returns:
        Configured registry
    """
    registry = get_global_registry()
    
    # Try to register Qiskit Aer (local simulator)
    try:
        from .qiskit_adapter import QiskitAdapter, QuantumJobConfig
        
        config = QuantumJobConfig(backend_name="qasm_simulator")
        adapter = QiskitAdapter(config)
        
        registry.register_adapter(
            "qiskit_aer",
            adapter,
            BackendCapabilities(
                max_qubits=32,
                is_simulator=True,
                cost_per_shot=0.0
            )
        )
        logger.info("Registered Qiskit Aer simulator")
    except Exception as e:
        logger.warning(f"Failed to register Qiskit Aer: {e}")
    
    # Try to register Azure Quantum
    try:
        from .azure_adapter import AzureAdapter, AzureQuantumConfig
        
        config = AzureQuantumConfig()
        
        # Only register if Azure credentials are available
        if config.subscription_id and config.resource_group and config.workspace_name:
            adapter = AzureAdapter(config)
            
            registry.register_adapter(
                "azure_quantum",
                adapter,
                BackendCapabilities(
                    max_qubits=29,  # IonQ typical
                    is_simulator=False,
                    cost_per_shot=0.0001
                )
            )
            logger.info("Registered Azure Quantum")
    except Exception as e:
        logger.debug(f"Azure Quantum not available: {e}")
    
    return registry
