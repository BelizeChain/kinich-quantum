"""
SpinQ Adapter for Kinich

Provides integration with SpinQ quantum computers for
community-operated quantum nodes in BelizeChain.

SpinQ offers affordable, desktop quantum computers perfect
for educational institutions and community participants.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SpinQConfig:
    """Configuration for SpinQ quantum computer."""
    
    # SpinQ API settings
    api_endpoint: str = field(default="http://localhost:8080")
    api_key: Optional[str] = field(default=None)
    
    # Device settings
    device_type: str = field(default="SpinQ-Gemini")  # or "SpinQ-Triangulum"
    max_qubits: int = field(default=2)  # Gemini: 2, Triangulum: 3
    
    # Execution settings
    default_shots: int = field(default=1024)
    max_shots: int = field(default=8192)
    timeout_seconds: int = field(default=300)
    
    # Local simulator fallback
    use_simulator_fallback: bool = field(default=True)
    
    # Community settings
    node_name: str = field(default="spinq-community-node")
    operator_address: Optional[str] = field(default=None)  # Blockchain address
    share_resources: bool = field(default=True)
    
    def __post_init__(self):
        """Load configuration from environment."""
        self.api_endpoint = os.getenv("SPINQ_API_ENDPOINT", self.api_endpoint)
        self.api_key = os.getenv("SPINQ_API_KEY", self.api_key)
        self.device_type = os.getenv("SPINQ_DEVICE_TYPE", self.device_type)
        self.node_name = os.getenv("KINICH_NODE_NAME", self.node_name)
        self.operator_address = os.getenv("BELIZECHAIN_OPERATOR_ADDRESS", self.operator_address)
        
        # Set max qubits based on device type
        if "Gemini" in self.device_type:
            self.max_qubits = 2
        elif "Triangulum" in self.device_type:
            self.max_qubits = 3
        else:
            logger.warning(f"Unknown SpinQ device: {self.device_type}, defaulting to 2 qubits")
            self.max_qubits = 2


class SpinQAdapter:
    """
    Adapter for SpinQ quantum computers.
    
    SpinQ provides affordable desktop quantum computers ideal for:
    - Educational institutions in Belize
    - Community quantum nodes
    - Local research projects
    - Quantum computing education
    
    Supported devices:
    - SpinQ Gemini: 2-qubit desktop quantum computer
    - SpinQ Triangulum: 3-qubit desktop quantum computer
    """
    
    def __init__(self, config: Optional[SpinQConfig] = None):
        """
        Initialize SpinQ adapter.
        
        Args:
            config: SpinQ configuration
        """
        self.config = config or SpinQConfig()
        self._device = None
        self._simulator = None
        self._connected = False
        self._job_history: List[Dict[str, Any]] = []
        
        logger.info(
            f"Initialized SpinQ adapter: {self.config.device_type} "
            f"({self.config.max_qubits} qubits)"
        )
        
        # Try to connect
        try:
            self._connect()
        except Exception as e:
            logger.warning(f"SpinQ device not available: {e}")
            
            if self.config.use_simulator_fallback:
                logger.info("Falling back to local simulator")
                self._initialize_simulator()
    
    def _connect(self) -> None:
        """Connect to SpinQ device."""
        try:
            # Try importing SpinQ SDK
            # Note: This is a placeholder - actual SpinQ SDK may differ
            import spinq
            
            self._device = spinq.connect(
                endpoint=self.config.api_endpoint,
                api_key=self.config.api_key,
                device_type=self.config.device_type
            )
            
            self._connected = True
            logger.info(f"Connected to SpinQ device at {self.config.api_endpoint}")
            
        except ImportError:
            logger.warning("SpinQ SDK not installed (pip install spinq-sdk)")
            raise RuntimeError("SpinQ SDK not available")
        except Exception as e:
            logger.error(f"Failed to connect to SpinQ device: {e}")
            raise
    
    def _initialize_simulator(self) -> None:
        """Initialize local Qiskit simulator as fallback."""
        try:
            from qiskit_aer import AerSimulator
            
            self._simulator = AerSimulator(method='statevector')
            logger.info("Initialized local simulator fallback")
            
        except ImportError:
            logger.error("Qiskit Aer not available for simulator fallback")
    
    def is_connected(self) -> bool:
        """Check if connected to SpinQ device."""
        return self._connected
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get SpinQ device information.
        
        Returns:
            Device specifications
        """
        return {
            'device_type': self.config.device_type,
            'max_qubits': self.config.max_qubits,
            'connected': self._connected,
            'api_endpoint': self.config.api_endpoint,
            'node_name': self.config.node_name,
            'operator_address': self.config.operator_address,
            'share_resources': self.config.share_resources,
            'using_simulator': self._simulator is not None and not self._connected,
            'capabilities': {
                'supports_measurement': True,
                'supports_reset': True,
                'supports_conditional': False,
                'basis_gates': ['h', 'x', 'y', 'z', 'cx', 'rx', 'ry', 'rz'],
            }
        }
    
    def execute_circuit(
        self,
        circuit: Any,
        shots: Optional[int] = None,
        job_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute quantum circuit on SpinQ device.
        
        Args:
            circuit: Qiskit QuantumCircuit
            shots: Number of measurements
            job_name: Job identifier
        
        Returns:
            Execution results
        """
        if shots is None:
            shots = self.config.default_shots
        
        shots = min(shots, self.config.max_shots)
        
        job_id = job_name or f"spinq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(
            f"Executing circuit on SpinQ: {job_id}, "
            f"{circuit.num_qubits} qubits, {shots} shots"
        )
        
        # Check qubit limit
        if circuit.num_qubits > self.config.max_qubits:
            error_msg = (
                f"Circuit requires {circuit.num_qubits} qubits, "
                f"but {self.config.device_type} only has {self.config.max_qubits}"
            )
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'job_id': job_id,
            }
        
        try:
            # Execute on actual device if connected
            if self._connected and self._device is not None:
                result = self._execute_on_device(circuit, shots, job_id)
            
            # Otherwise use simulator
            elif self._simulator is not None:
                logger.info(f"Executing {job_id} on local simulator")
                result = self._execute_on_simulator(circuit, shots, job_id)
            
            else:
                return {
                    'success': False,
                    'error': 'No SpinQ device or simulator available',
                    'job_id': job_id,
                }
            
            # Record job
            self._job_history.append({
                'job_id': job_id,
                'timestamp': datetime.now().isoformat(),
                'qubits': circuit.num_qubits,
                'shots': shots,
                'success': result.get('success', False),
                'device': self.config.device_type if self._connected else 'simulator',
            })
            
            return result
            
        except Exception as e:
            logger.error(f"SpinQ execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'job_id': job_id,
            }
    
    def _execute_on_device(
        self,
        circuit: Any,
        shots: int,
        job_id: str
    ) -> Dict[str, Any]:
        """Execute on actual SpinQ hardware."""
        # Convert Qiskit circuit to SpinQ format
        spinq_circuit = self._translate_circuit(circuit)
        
        # Submit to SpinQ device
        job = self._device.run(spinq_circuit, shots=shots)
        
        # Wait for completion
        result = job.result(timeout=self.config.timeout_seconds)
        
        # Extract counts
        counts = result.get_counts()
        
        return {
            'success': True,
            'job_id': job_id,
            'counts': counts,
            'shots': shots,
            'backend': self.config.device_type,
            'execution_time': result.execution_time if hasattr(result, 'execution_time') else None,
            'device_type': 'hardware',
            'estimated_cost': 0.0,  # Community nodes typically free
        }
    
    def _execute_on_simulator(
        self,
        circuit: Any,
        shots: int,
        job_id: str
    ) -> Dict[str, Any]:
        """Execute on local Qiskit simulator."""
        from qiskit import transpile
        
        # Transpile for simulator
        transpiled = transpile(circuit, self._simulator)
        
        # Execute
        job = self._simulator.run(transpiled, shots=shots)
        result = job.result()
        
        # Get counts
        counts = result.get_counts()
        
        return {
            'success': True,
            'job_id': job_id,
            'counts': counts,
            'shots': shots,
            'backend': 'local_simulator',
            'device_type': 'simulator',
            'estimated_cost': 0.0,
        }
    
    def _translate_circuit(self, qiskit_circuit: Any) -> Any:
        """
        Translate Qiskit circuit to SpinQ format.
        
        Args:
            qiskit_circuit: Qiskit QuantumCircuit
        
        Returns:
            SpinQ circuit
        """
        # Placeholder for SpinQ circuit translation
        # Actual implementation depends on SpinQ SDK
        
        # For now, assume SpinQ accepts OpenQASM
        qasm = qiskit_circuit.qasm()
        return qasm
    
    def submit_qasm_job(
        self,
        qasm_string: str,
        shots: Optional[int] = None,
        job_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit OpenQASM 2.0 job to SpinQ.
        
        Args:
            qasm_string: OpenQASM code
            shots: Number of shots
            job_name: Job identifier
        
        Returns:
            Execution results
        """
        # Convert QASM to Qiskit circuit
        from qiskit import QuantumCircuit
        
        circuit = QuantumCircuit.from_qasm_str(qasm_string)
        
        return self.execute_circuit(circuit, shots, job_name)
    
    def get_job_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent job history."""
        return self._job_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get node statistics."""
        total_jobs = len(self._job_history)
        successful_jobs = sum(1 for job in self._job_history if job['success'])
        
        total_shots = sum(job['shots'] for job in self._job_history)
        
        return {
            'node_name': self.config.node_name,
            'operator_address': self.config.operator_address,
            'device_type': self.config.device_type,
            'total_jobs': total_jobs,
            'successful_jobs': successful_jobs,
            'success_rate': successful_jobs / total_jobs if total_jobs > 0 else 0.0,
            'total_shots': total_shots,
            'connected': self._connected,
            'using_simulator': self._simulator is not None and not self._connected,
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'node_name': self.config.node_name,
            'device_type': self.config.device_type,
            'connected': self._connected,
            'operational': self._connected or (self._simulator is not None),
        }
        
        # Test execution
        try:
            from qiskit import QuantumCircuit
            
            # Simple 1-qubit circuit
            test_circuit = QuantumCircuit(1, 1)
            test_circuit.h(0)
            test_circuit.measure(0, 0)
            
            result = self.execute_circuit(test_circuit, shots=10, job_name="health_check")
            
            health['test_execution'] = result.get('success', False)
            health['status'] = 'healthy' if result.get('success') else 'degraded'
            
        except Exception as e:
            health['test_execution'] = False
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health


# Utility functions
def create_spinq_adapter_from_env() -> SpinQAdapter:
    """Create SpinQ adapter from environment variables."""
    config = SpinQConfig()
    return SpinQAdapter(config)


def is_spinq_available() -> bool:
    """Check if SpinQ SDK is available."""
    try:
        import spinq
        return True
    except ImportError:
        return False
