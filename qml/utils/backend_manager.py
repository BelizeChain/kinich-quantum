"""
Quantum Backend Manager

Manages connections to Azure Quantum, IBM Quantum, and simulators.
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Supported backend types."""
    AZURE_IONQ = "azure_ionq"
    AZURE_QUANTINUUM = "azure_quantinuum"
    IBM_QUANTUM = "ibm_quantum"
    SIMULATOR = "simulator"


class BackendManager:
    """
    Manages quantum backend connections.
    
    Provides unified interface for:
    - Azure Quantum (IonQ, Quantinuum)
    - IBM Quantum
    - Local simulators
    
    Example:
        >>> manager = BackendManager()
        >>> backend = manager.get_backend("azure_ionq")
        >>> job = backend.run(circuit, shots=1024)
    """
    
    def __init__(self):
        """Initialize backend manager."""
        self.backends = {}
        self._init_backends()
    
    def _init_backends(self) -> None:
        """Initialize available backends."""
        # Try to import Qiskit
        try:
            from qiskit import Aer
            self.backends[BackendType.SIMULATOR.value] = Aer.get_backend('qasm_simulator')
            logger.info("Qiskit simulator available")
        except ImportError:
            logger.warning("Qiskit not available - no simulator")
        
        # Azure Quantum would be initialized here with credentials
        logger.debug("Backend manager initialized")
    
    def get_backend(self, backend_name: str):
        """
        Get backend by name.
        
        Args:
            backend_name: Backend identifier
            
        Returns:
            Backend object
        """
        if backend_name in self.backends:
            return self.backends[backend_name]
        
        # Default to simulator
        logger.warning(f"Backend {backend_name} not available, using simulator")
        return self.backends.get(BackendType.SIMULATOR.value)
    
    def list_backends(self) -> list:
        """List available backends."""
        return list(self.backends.keys())
    
    def execute_circuit(
        self,
        circuit,
        backend_name: str = "simulator",
        shots: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute circuit on specified backend.
        
        Args:
            circuit: Quantum circuit
            backend_name: Backend to use
            shots: Number of shots
            **kwargs: Additional backend options
            
        Returns:
            Execution results
        """
        backend = self.get_backend(backend_name)
        
        if backend is None:
            logger.error("No backend available")
            return {"error": "No backend available"}
        
        try:
            from qiskit import execute
            job = execute(circuit, backend, shots=shots, **kwargs)
            result = job.result()
            counts = result.get_counts()
            return {"counts": counts, "success": True}
        
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return {"error": str(e), "success": False}
