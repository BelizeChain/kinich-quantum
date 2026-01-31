"""
Azure Quantum Adapter for Kinich

Primary production quantum computing backend using Azure Quantum.
Supports multiple quantum hardware providers through Azure:
- IonQ
- Quantinuum (Honeywell)
- Rigetti
- Quantum Circuits Inc (QCI)

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import os
import logging
from datetime import datetime

# Azure Quantum imports
try:
    from azure.quantum import Workspace
    from azure.quantum.qiskit import AzureQuantumProvider
    from azure.quantum.cirq import AzureQuantumService
    from azure.identity import DefaultAzureCredential, ClientSecretCredential
    from azure.core.exceptions import AzureError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logging.warning("Azure Quantum SDK not available. Install: pip install azure-quantum")

# Qiskit for circuit manipulation
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.result import Result
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AzureQuantumConfig:
    """Configuration for Azure Quantum connection."""
    
    # Azure credentials
    subscription_id: Optional[str] = None
    resource_group: Optional[str] = None
    workspace_name: Optional[str] = None
    location: str = "eastus"
    
    # Authentication
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    use_managed_identity: bool = False
    
    # Connection string (alternative to individual params)
    connection_string: Optional[str] = None
    
    # Backend selection
    default_backend: str = "ionq.simulator"  # ionq.simulator, ionq.qpu, quantinuum, rigetti
    
    # Job configuration
    default_shots: int = 1024
    max_shots: int = 10000
    timeout_seconds: int = 600
    
    # Cost management
    max_cost_per_job: Optional[float] = None  # Maximum cost in USD
    track_costs: bool = True
    
    def __post_init__(self):
        """Load from environment if not provided."""
        if self.subscription_id is None:
            self.subscription_id = os.getenv('AZURE_QUANTUM_SUBSCRIPTION_ID')
        if self.resource_group is None:
            self.resource_group = os.getenv('AZURE_QUANTUM_RESOURCE_GROUP')
        if self.workspace_name is None:
            self.workspace_name = os.getenv('AZURE_QUANTUM_WORKSPACE_NAME')
        if self.location is None:
            self.location = os.getenv('AZURE_QUANTUM_LOCATION', 'eastus')
        
        # Authentication
        if self.tenant_id is None:
            self.tenant_id = os.getenv('AZURE_TENANT_ID')
        if self.client_id is None:
            self.client_id = os.getenv('AZURE_CLIENT_ID')
        if self.client_secret is None:
            self.client_secret = os.getenv('AZURE_CLIENT_SECRET')


class AzureAdapter:
    """
    Azure Quantum adapter for Kinich distributed quantum computing.
    
    Provides access to multiple quantum hardware providers through
    Azure Quantum workspace.
    """
    
    def __init__(self, config: Optional[AzureQuantumConfig] = None):
        """
        Initialize Azure Quantum adapter.
        
        Args:
            config: Azure Quantum configuration (loads from env if None)
        """
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure Quantum SDK required. Install: pip install azure-quantum qiskit-ionq"
            )
        
        self.config = config or AzureQuantumConfig()
        self.workspace: Optional[Workspace] = None
        self.provider: Optional[AzureQuantumProvider] = None
        self.credential: Optional[Any] = None
        
        # Available backends cache
        self._available_backends: Dict[str, Any] = {}
        self._backend_info_cache: Dict[str, Dict[str, Any]] = {}
        
        # Cost tracking
        self._total_cost = 0.0
        self._job_costs: Dict[str, float] = {}
        
        self._connect()
        
        logger.info(f"Initialized Azure Quantum adapter: {self.config.workspace_name}")
    
    def _get_credential(self) -> Any:
        """Get Azure credential for authentication."""
        if self.credential:
            return self.credential
        
        if self.config.use_managed_identity:
            # Use managed identity (for Azure VMs, App Service, etc.)
            self.credential = DefaultAzureCredential()
            logger.info("Using Azure Managed Identity for authentication")
        
        elif (self.config.tenant_id and 
              self.config.client_id and 
              self.config.client_secret):
            # Use service principal
            self.credential = ClientSecretCredential(
                tenant_id=self.config.tenant_id,
                client_id=self.config.client_id,
                client_secret=self.config.client_secret
            )
            logger.info("Using Azure Service Principal for authentication")
        
        else:
            # Use default Azure credential chain
            self.credential = DefaultAzureCredential()
            logger.info("Using Default Azure Credential chain")
        
        return self.credential
    
    def _connect(self) -> None:
        """Connect to Azure Quantum workspace."""
        try:
            if self.config.connection_string:
                # Connect using connection string
                self.workspace = Workspace.from_connection_string(
                    self.config.connection_string
                )
                logger.info("Connected to Azure Quantum using connection string")
            
            else:
                # Connect using workspace parameters
                credential = self._get_credential()
                
                self.workspace = Workspace(
                    subscription_id=self.config.subscription_id,
                    resource_group=self.config.resource_group,
                    name=self.config.workspace_name,
                    location=self.config.location,
                    credential=credential
                )
                logger.info(
                    f"Connected to Azure Quantum workspace: "
                    f"{self.config.workspace_name} in {self.config.location}"
                )
            
            # Initialize Qiskit provider
            if QISKIT_AVAILABLE:
                self.provider = AzureQuantumProvider(self.workspace)
                logger.info("Initialized Qiskit provider for Azure Quantum")
            
            # Cache available backends
            self._refresh_backends()
            
        except AzureError as e:
            logger.error(f"Failed to connect to Azure Quantum: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Azure Quantum: {e}")
            raise
    
    def _refresh_backends(self) -> None:
        """Refresh list of available backends."""
        if not self.provider:
            return
        
        try:
            backends = self.provider.backends()
            self._available_backends = {b.name(): b for b in backends}
            
            logger.info(f"Found {len(self._available_backends)} available backends")
            
            # Log backend details
            for name, backend in self._available_backends.items():
                logger.debug(f"  - {name}: {backend.status()}")
        
        except Exception as e:
            logger.warning(f"Failed to refresh backends: {e}")
    
    def list_backends(self) -> List[str]:
        """
        List available quantum backends.
        
        Returns:
            List of backend names
        """
        return list(self._available_backends.keys())
    
    def get_backend_info(self, backend_name: str) -> Dict[str, Any]:
        """
        Get information about a specific backend.
        
        Args:
            backend_name: Name of the backend
        
        Returns:
            Backend information dictionary
        """
        if backend_name in self._backend_info_cache:
            return self._backend_info_cache[backend_name]
        
        if backend_name not in self._available_backends:
            return {"error": f"Backend '{backend_name}' not available"}
        
        backend = self._available_backends[backend_name]
        
        try:
            config = backend.configuration()
            status = backend.status()
            
            info = {
                "name": backend_name,
                "provider": getattr(config, "provider", "unknown"),
                "num_qubits": getattr(config, "n_qubits", None),
                "basis_gates": getattr(config, "basis_gates", []),
                "coupling_map": getattr(config, "coupling_map", None),
                "is_simulator": getattr(config, "simulator", False),
                "max_shots": getattr(config, "max_shots", None),
                "max_experiments": getattr(config, "max_experiments", 1),
                "status": {
                    "operational": status.operational if hasattr(status, "operational") else True,
                    "pending_jobs": status.pending_jobs if hasattr(status, "pending_jobs") else 0,
                    "message": status.status_msg if hasattr(status, "status_msg") else "Available",
                },
            }
            
            self._backend_info_cache[backend_name] = info
            return info
        
        except Exception as e:
            logger.error(f"Failed to get backend info for {backend_name}: {e}")
            return {"error": str(e)}
    
    def execute_circuit(
        self,
        circuit: QuantumCircuit,
        backend_name: Optional[str] = None,
        shots: Optional[int] = None,
        job_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a quantum circuit on Azure Quantum.
        
        Args:
            circuit: Qiskit quantum circuit
            backend_name: Backend to use (uses default if None)
            shots: Number of shots (uses config default if None)
            job_name: Optional job name for tracking
        
        Returns:
            Execution results dictionary
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for circuit execution")
        
        # Select backend
        backend_name = backend_name or self.config.default_backend
        if backend_name not in self._available_backends:
            raise ValueError(f"Backend '{backend_name}' not available")
        
        backend = self._available_backends[backend_name]
        
        # Set shots
        shots = shots or self.config.default_shots
        shots = min(shots, self.config.max_shots)
        
        # Generate job name
        if job_name is None:
            job_name = f"kinich_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Executing circuit on {backend_name} with {shots} shots")
        
        try:
            # Transpile circuit for backend
            transpiled = transpile(circuit, backend=backend, optimization_level=2)
            
            # Submit job
            job = backend.run(transpiled, shots=shots, job_name=job_name)
            
            logger.info(f"Job submitted: {job.job_id()}")
            
            # Wait for result
            result = job.result(timeout=self.config.timeout_seconds)
            
            # Extract results
            counts = result.get_counts()
            
            # Estimate cost (if tracking enabled)
            estimated_cost = self._estimate_cost(backend_name, shots, circuit.depth())
            if self.config.track_costs:
                self._job_costs[job.job_id()] = estimated_cost
                self._total_cost += estimated_cost
            
            execution_result = {
                "job_id": job.job_id(),
                "backend": backend_name,
                "shots": shots,
                "counts": counts,
                "success": result.success,
                "execution_time": getattr(result, "time_taken", None),
                "estimated_cost": estimated_cost if self.config.track_costs else None,
                "circuit_depth": circuit.depth(),
                "circuit_width": circuit.num_qubits,
            }
            
            logger.info(f"Job completed successfully: {job.job_id()}")
            return execution_result
        
        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            return {
                "error": str(e),
                "backend": backend_name,
                "shots": shots,
                "success": False,
            }
    
    def _estimate_cost(self, backend_name: str, shots: int, depth: int) -> float:
        """
        Estimate job cost in USD.
        
        This is a rough estimate based on typical Azure Quantum pricing.
        Actual costs may vary.
        
        Args:
            backend_name: Backend name
            shots: Number of shots
            depth: Circuit depth
        
        Returns:
            Estimated cost in USD
        """
        # Pricing estimates (as of 2024, subject to change)
        # These are approximate and should be updated with actual pricing
        
        if "simulator" in backend_name.lower():
            # Simulators are typically free or very low cost
            return 0.0
        
        elif "ionq" in backend_name.lower():
            # IonQ pricing: ~$0.00003 per gate-shot
            gates = depth * shots
            return gates * 0.00003
        
        elif "quantinuum" in backend_name.lower() or "honeywell" in backend_name.lower():
            # Quantinuum pricing: ~$0.0001 per circuit shot
            return shots * 0.0001
        
        elif "rigetti" in backend_name.lower():
            # Rigetti pricing: ~$0.00002 per gate-shot
            gates = depth * shots
            return gates * 0.00002
        
        else:
            # Default estimate
            return shots * 0.0001
    
    def get_total_cost(self) -> float:
        """Get total cost of all jobs executed."""
        return self._total_cost
    
    def get_job_cost(self, job_id: str) -> Optional[float]:
        """Get cost of a specific job."""
        return self._job_costs.get(job_id)
    
    def submit_qasm_job(
        self,
        qasm_string: str,
        backend_name: Optional[str] = None,
        shots: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Submit QASM circuit for execution.
        
        Args:
            qasm_string: OpenQASM 2.0 string
            backend_name: Backend to use
            shots: Number of shots
        
        Returns:
            Execution results
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required")
        
        try:
            # Parse QASM to Qiskit circuit
            circuit = QuantumCircuit.from_qasm_str(qasm_string)
            
            # Execute
            return self.execute_circuit(circuit, backend_name, shots)
        
        except Exception as e:
            logger.error(f"QASM job submission failed: {e}")
            return {"error": str(e), "success": False}
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get Azure Quantum workspace information."""
        if not self.workspace:
            return {"error": "Not connected to workspace"}
        
        return {
            "name": self.config.workspace_name,
            "location": self.config.location,
            "resource_group": self.config.resource_group,
            "subscription_id": self.config.subscription_id,
            "available_backends": len(self._available_backends),
            "total_cost": self._total_cost if self.config.track_costs else None,
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Azure Quantum connection.
        
        Returns:
            Health status dictionary
        """
        health = {
            "connected": self.workspace is not None,
            "provider_available": self.provider is not None,
            "backends_available": len(self._available_backends),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Test backend connectivity
        if self._available_backends:
            backend_name = list(self._available_backends.keys())[0]
            try:
                backend = self._available_backends[backend_name]
                status = backend.status()
                health["backend_operational"] = getattr(status, "operational", True)
            except Exception as e:
                health["backend_operational"] = False
                health["backend_error"] = str(e)
        
        return health


# Utility functions

def create_azure_adapter_from_env() -> AzureAdapter:
    """
    Create Azure Quantum adapter from environment variables.
    
    Required environment variables:
    - AZURE_QUANTUM_SUBSCRIPTION_ID
    - AZURE_QUANTUM_RESOURCE_GROUP
    - AZURE_QUANTUM_WORKSPACE_NAME
    
    Optional:
    - AZURE_QUANTUM_LOCATION (default: eastus)
    - AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET (for service principal auth)
    
    Returns:
        Configured AzureAdapter instance
    """
    config = AzureQuantumConfig()  # Will load from environment
    return AzureAdapter(config)


def list_azure_backends() -> List[str]:
    """
    Quick utility to list available Azure Quantum backends.
    
    Returns:
        List of backend names
    """
    try:
        adapter = create_azure_adapter_from_env()
        return adapter.list_backends()
    except Exception as e:
        logger.error(f"Failed to list backends: {e}")
        return []
