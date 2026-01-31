"""
Job Factories and Handlers for Kinich

Factory methods and handlers to create and process quantum jobs.
Integrates circuit builders with job execution pipeline.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, Optional, Any
import logging

from ..core import (
    QuantumJob,
    CryptographyJob,
    OptimizationJob,
    SimulationJob,
    AIEnhancementJob,
    ConsensusJob,
    JobType,
    JobPriority,
)
from ..circuits import CircuitBuilder, get_required_qubits

logger = logging.getLogger(__name__)


class JobFactory:
    """
    Factory for creating quantum jobs from specifications.
    
    Converts job requests into executable QuantumJob objects
    with appropriate parameters and metadata.
    """
    
    @staticmethod
    def create_job(
        job_type: str,
        priority: str = "NORMAL",
        **kwargs
    ) -> Optional[QuantumJob]:
        """
        Create quantum job from specification.
        
        Args:
            job_type: Type of quantum job (JobType enum value)
            priority: Job priority (JobPriority enum value)
            **kwargs: Job-specific parameters
        
        Returns:
            QuantumJob instance or None
        """
        try:
            job_type_enum = JobType[job_type]
            priority_enum = JobPriority[priority]
        except KeyError as e:
            logger.error(f"Invalid job type or priority: {e}")
            return None
        
        # Route to specific factory based on job category
        if job_type_enum in [
            JobType.QUANTUM_KEY_GENERATION,
            JobType.POST_QUANTUM_CRYPTO,
            JobType.QUANTUM_RANDOM,
            JobType.HASH_VERIFICATION,
        ]:
            return JobFactory._create_cryptography_job(job_type_enum, priority_enum, **kwargs)
        
        elif job_type_enum in [
            JobType.QAOA_OPTIMIZATION,
            JobType.VQE_OPTIMIZATION,
            JobType.PORTFOLIO_OPTIMIZATION,
            JobType.ROUTE_OPTIMIZATION,
        ]:
            return JobFactory._create_optimization_job(job_type_enum, priority_enum, **kwargs)
        
        elif job_type_enum in [
            JobType.QUANTUM_CHEMISTRY,
            JobType.MATERIAL_SIMULATION,
            JobType.MOLECULAR_DYNAMICS,
        ]:
            return JobFactory._create_simulation_job(job_type_enum, priority_enum, **kwargs)
        
        elif job_type_enum in [
            JobType.QUANTUM_NEURAL_NETWORK,
            JobType.QUANTUM_KERNEL,
            JobType.FEDERATED_LEARNING_AGGREGATION,
            JobType.VARIATIONAL_CLASSIFIER,
        ]:
            return JobFactory._create_ai_job(job_type_enum, priority_enum, **kwargs)
        
        elif job_type_enum in [
            JobType.CONSENSUS_VERIFICATION,
            JobType.BLOCK_VALIDATION,
            JobType.TRANSACTION_ORDERING,
        ]:
            return JobFactory._create_consensus_job(job_type_enum, priority_enum, **kwargs)
        
        else:
            # Generic job
            return QuantumJob(
                job_type=job_type_enum,
                priority=priority_enum,
                **kwargs
            )
    
    @staticmethod
    def _create_cryptography_job(
        job_type: JobType,
        priority: JobPriority,
        **kwargs
    ) -> CryptographyJob:
        """Create cryptography job."""
        return CryptographyJob(
            job_type=job_type,
            priority=priority,
            crypto_operation=kwargs.get('crypto_operation', 'key_generation'),
            key_length=kwargs.get('key_length', 256),
            algorithm=kwargs.get('algorithm', 'bb84'),
            shots=kwargs.get('shots', 1024),
            backend_preference=kwargs.get('backend_preference'),
            originator=kwargs.get('originator'),
            originating_pallet=kwargs.get('originating_pallet', 'pallet-economy'),
        )
    
    @staticmethod
    def _create_optimization_job(
        job_type: JobType,
        priority: JobPriority,
        **kwargs
    ) -> OptimizationJob:
        """Create optimization job."""
        return OptimizationJob(
            job_type=job_type,
            priority=priority,
            algorithm=kwargs.get('algorithm', 'QAOA'),
            cost_function=kwargs.get('cost_function'),
            constraints=kwargs.get('constraints', {}),
            max_iterations=kwargs.get('max_iterations', 100),
            shots=kwargs.get('shots', 2048),
            backend_preference=kwargs.get('backend_preference'),
            originator=kwargs.get('originator'),
            originating_pallet=kwargs.get('originating_pallet', 'pallet-economy'),
        )
    
    @staticmethod
    def _create_simulation_job(
        job_type: JobType,
        priority: JobPriority,
        **kwargs
    ) -> SimulationJob:
        """Create simulation job."""
        return SimulationJob(
            job_type=job_type,
            priority=priority,
            simulation_type=kwargs.get('simulation_type', 'molecular'),
            molecule=kwargs.get('molecule', 'H2'),
            hamiltonian=kwargs.get('hamiltonian'),
            basis_set=kwargs.get('basis_set', 'sto-3g'),
            shots=kwargs.get('shots', 4096),
            backend_preference=kwargs.get('backend_preference'),
            originator=kwargs.get('originator'),
            originating_pallet=kwargs.get('originating_pallet', 'pallet-research'),
        )
    
    @staticmethod
    def _create_ai_job(
        job_type: JobType,
        priority: JobPriority,
        **kwargs
    ) -> AIEnhancementJob:
        """Create AI enhancement job."""
        return AIEnhancementJob(
            job_type=job_type,
            priority=priority,
            ai_operation=kwargs.get('ai_operation', 'classification'),
            model_parameters=kwargs.get('model_parameters', {}),
            feature_dimension=kwargs.get('feature_dimension', 4),
            training_data=kwargs.get('training_data'),
            shots=kwargs.get('shots', 2048),
            backend_preference=kwargs.get('backend_preference'),
            originator=kwargs.get('originator'),
            originating_pallet=kwargs.get('originating_pallet', 'pallet-ai'),
        )
    
    @staticmethod
    def _create_consensus_job(
        job_type: JobType,
        priority: JobPriority,
        **kwargs
    ) -> ConsensusJob:
        """Create consensus job."""
        return ConsensusJob(
            job_type=job_type,
            priority=priority,
            consensus_operation=kwargs.get('consensus_operation', 'verification'),
            validator_votes=kwargs.get('validator_votes', []),
            block_data=kwargs.get('block_data'),
            threshold=kwargs.get('threshold', 0.67),
            shots=kwargs.get('shots', 1024),
            backend_preference=kwargs.get('backend_preference'),
            originator=kwargs.get('originator'),
            originating_pallet=kwargs.get('originating_pallet', 'pallet-consensus'),
        )


class JobHandler:
    """
    Handles quantum job execution pipeline.
    
    Integrates circuit building, optimization, execution,
    and result aggregation.
    """
    
    def __init__(
        self,
        circuit_optimizer=None,
        result_aggregator=None,
        adapter_registry=None
    ):
        """
        Initialize job handler.
        
        Args:
            circuit_optimizer: CircuitOptimizer instance
            result_aggregator: ResultAggregator instance
            adapter_registry: AdapterRegistry instance
        """
        self.circuit_optimizer = circuit_optimizer
        self.result_aggregator = result_aggregator
        self.adapter_registry = adapter_registry
        
        logger.info("Initialized job handler")
    
    def process_job(self, job: QuantumJob) -> Dict[str, Any]:
        """
        Process quantum job end-to-end.
        
        Args:
            job: Quantum job to process
        
        Returns:
            Job result dictionary
        """
        logger.info(f"Processing job {job.get_job_id()}")
        
        try:
            # 1. Build circuit
            circuit = self._build_circuit_for_job(job)
            
            if circuit is None:
                return {
                    'success': False,
                    'error': 'Failed to build circuit',
                    'job_id': job.get_job_id(),
                }
            
            # 2. Optimize circuit
            if self.circuit_optimizer:
                backend = self._select_backend(job)
                circuit = self.circuit_optimizer.optimize_for_backend(
                    circuit,
                    backend
                )
            
            # 3. Execute circuit
            result = self._execute_circuit(circuit, job)
            
            return result
            
        except Exception as e:
            logger.error(f"Job processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'job_id': job.get_job_id(),
            }
    
    def _build_circuit_for_job(self, job: QuantumJob) -> Optional[Any]:
        """Build quantum circuit for job."""
        job_params = self._extract_job_parameters(job)
        
        circuit = CircuitBuilder.build_circuit(
            job.job_type.name,
            **job_params
        )
        
        if circuit:
            logger.info(f"Built circuit for {job.job_type.name}")
        
        return circuit
    
    def _extract_job_parameters(self, job: QuantumJob) -> Dict[str, Any]:
        """Extract parameters from job for circuit building."""
        params = {}
        
        # Common parameters
        if hasattr(job, 'shots'):
            params['shots'] = job.shots
        
        # Cryptography parameters
        if isinstance(job, CryptographyJob):
            params['key_length'] = job.key_length
            params['algorithm'] = job.algorithm
        
        # Optimization parameters
        elif isinstance(job, OptimizationJob):
            params['algorithm'] = job.algorithm
            params['cost_function'] = job.cost_function
            params['constraints'] = job.constraints
            params['max_iterations'] = job.max_iterations
        
        # Simulation parameters
        elif isinstance(job, SimulationJob):
            params['molecule'] = job.molecule
            params['hamiltonian'] = job.hamiltonian
            params['basis_set'] = job.basis_set
        
        # AI parameters
        elif isinstance(job, AIEnhancementJob):
            params['feature_dim'] = job.feature_dimension
            params['model_params'] = job.model_parameters
            params['layers'] = 2  # Default
        
        # Consensus parameters
        elif isinstance(job, ConsensusJob):
            params['validators'] = len(job.validator_votes) if job.validator_votes else 8
            params['threshold'] = job.threshold
        
        return params
    
    def _select_backend(self, job: QuantumJob) -> str:
        """Select best backend for job."""
        if self.adapter_registry:
            # Get qubit requirement
            min_qubits = get_required_qubits(
                job.job_type.name,
                **self._extract_job_parameters(job)
            )
            
            # Find suitable adapter
            backend = self.adapter_registry.find_suitable_adapter(
                min_qubits=min_qubits,
                prefer_simulator=True  # Default to simulator
            )
            
            return backend or 'qiskit_aer'
        
        return 'qiskit_aer'
    
    def _execute_circuit(
        self,
        circuit: Any,
        job: QuantumJob
    ) -> Dict[str, Any]:
        """Execute circuit on quantum backend."""
        if self.adapter_registry:
            return self.adapter_registry.execute_on_best_adapter(
                circuit=circuit,
                shots=job.shots,
                backend_preference=job.backend_preference
            )
        
        # Fallback: local execution
        return self._execute_local(circuit, job.shots)
    
    def _execute_local(self, circuit: Any, shots: int) -> Dict[str, Any]:
        """Execute circuit locally."""
        try:
            from qiskit_aer import AerSimulator
            from qiskit import transpile
            
            simulator = AerSimulator()
            transpiled = transpile(circuit, simulator)
            result = simulator.run(transpiled, shots=shots).result()
            
            return {
                'success': True,
                'counts': result.get_counts(),
                'shots': shots,
                'backend': 'local_simulator',
            }
        except Exception as e:
            logger.error(f"Local execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
            }


# Convenience functions
def create_quantum_key_job(
    key_length: int = 256,
    priority: str = "HIGH",
    **kwargs
) -> Optional[CryptographyJob]:
    """Create quantum key generation job."""
    return JobFactory.create_job(
        "QUANTUM_KEY_GENERATION",
        priority=priority,
        key_length=key_length,
        **kwargs
    )


def create_optimization_job(
    algorithm: str = "QAOA",
    num_qubits: int = 4,
    priority: str = "NORMAL",
    **kwargs
) -> Optional[OptimizationJob]:
    """Create optimization job."""
    return JobFactory.create_job(
        f"{algorithm.upper()}_OPTIMIZATION",
        priority=priority,
        num_qubits=num_qubits,
        **kwargs
    )


def create_consensus_job(
    validators: int = 8,
    priority: str = "CRITICAL",
    **kwargs
) -> Optional[ConsensusJob]:
    """Create consensus verification job."""
    return JobFactory.create_job(
        "CONSENSUS_VERIFICATION",
        priority=priority,
        validators=validators,
        **kwargs
    )
