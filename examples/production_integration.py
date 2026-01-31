"""
Kinich Production Integration Guide

Quick reference for integrating production hardening components
with the Kinich quantum computing infrastructure.

Author: BelizeChain Team
"""

from kinich.core import QuantumNode, JobScheduler
from kinich.security import SecurityManager
from kinich.error_mitigation import QuantumErrorMitigator, ErrorMitigationConfig
from kinich.sovereignty import SovereigntyManager, DataClassification
from kinich.monitoring import MonitoringManager
from kinich.config import ConfigManager


class ProductionKinichNode:
    """
    Production-hardened Kinich quantum node with comprehensive security and monitoring.
    
    Example usage:
        node = ProductionKinichNode('config.yaml')
        
        # Submit authenticated job
        job_id = node.submit_job(
            user_id='user123',
            api_key='kinich_abc123...',
            circuit=circuit,
            backend='spinq_local',
            data_classification=DataClassification.INTERNAL
        )
        
        # Get results with error mitigation
        results = node.get_results(job_id)
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize production Kinich node.
        
        Args:
            config_path: Path to configuration file
        """
        # 1. Load configuration
        self.config = ConfigManager(config_path)
        
        # 2. Initialize security
        self.security = SecurityManager()
        
        # 3. Initialize error mitigation
        error_config = ErrorMitigationConfig(
            enable_readout_mitigation=self.config.get('error_mitigation.enable_readout_mitigation', True),
            enable_zero_noise_extrapolation=self.config.get('error_mitigation.enable_zero_noise_extrapolation', False),
            enable_dynamic_decoupling=self.config.get('error_mitigation.enable_dynamic_decoupling', False),
        )
        self.error_mitigator = QuantumErrorMitigator(error_config)
        
        # 4. Initialize sovereignty controls
        self.sovereignty = SovereigntyManager()
        
        # 5. Initialize monitoring
        self.monitoring = MonitoringManager()
        
        # 6. Initialize core quantum node
        self.quantum_node = QuantumNode(
            node_id=self.config.get('quantum.node_id', 'node_1'),
            # Add other quantum node config
        )
        
        print("✅ Production Kinich node initialized")
    
    def submit_job(
        self,
        user_id: str,
        api_key: str,
        circuit: any,
        backend: str,
        data_classification: DataClassification = DataClassification.INTERNAL,
        **kwargs
    ) -> str:
        """
        Submit quantum job with full production features.
        
        Steps:
        1. Authenticate user
        2. Authorize action (SUBMIT_JOB permission)
        3. Check data sovereignty compliance
        4. Validate circuit security
        5. Start monitoring span
        6. Submit to quantum node
        7. Record metrics
        
        Args:
            user_id: User ID
            api_key: API key for authentication
            circuit: Quantum circuit
            backend: Target backend
            data_classification: Data classification level
            **kwargs: Additional job parameters
        
        Returns:
            Job ID
        """
        # Start tracing span
        span_id = self.monitoring.start_span(
            "submit_job",
            attributes={
                'user_id': user_id,
                'backend': backend,
                'classification': data_classification.value
            }
        )
        
        try:
            # 1. Authenticate
            user = self.security.authenticate_api_key(api_key)
            if not user:
                self.monitoring.end_span(span_id, status="error")
                raise ValueError("Authentication failed")
            
            # 2. Authorize
            authorized = self.security.authorize_action(
                user_id=user.user_id,
                action="submit_job",
                resource=f"backend:{backend}",
                details={'backend': backend}
            )
            if not authorized:
                self.monitoring.end_span(span_id, status="error")
                raise PermissionError("Not authorized to submit jobs")
            
            # 3. Check data sovereignty
            if not self.sovereignty.check_data_residency(
                data_classification=data_classification,
                backend_name=backend,
                user_id=user_id
            ):
                self.monitoring.end_span(span_id, status="error")
                raise ValueError("Data sovereignty violation")
            
            # 4. Validate circuit security
            if not self.security.validate_circuit_complexity(circuit):
                self.monitoring.end_span(span_id, status="error")
                raise ValueError("Circuit exceeds complexity limits")
            
            # 5. Check error budget
            if not self.error_mitigator.check_error_budget(circuit, backend):
                self.monitoring.add_span_event(
                    span_id,
                    "error_budget_warning",
                    {'message': 'High error rate expected'}
                )
            
            # 6. Submit to quantum node
            self.monitoring.add_span_event(span_id, "submitting_to_backend")
            job_id = self.quantum_node.submit_job(
                circuit=circuit,
                backend=backend,
                **kwargs
            )
            
            # 7. Record metrics
            self.monitoring.increment_counter('jobs_submitted', labels={'backend': backend})
            
            # End span
            self.monitoring.end_span(span_id, status="success")
            
            return job_id
            
        except Exception as e:
            self.monitoring.end_span(span_id, status="error")
            self.monitoring.increment_counter('job_submission_errors')
            raise
    
    def get_results(self, job_id: str) -> dict:
        """
        Get job results with error mitigation applied.
        
        Args:
            job_id: Job ID
        
        Returns:
            Mitigated results
        """
        span_id = self.monitoring.start_span(
            "get_results",
            attributes={'job_id': job_id}
        )
        
        try:
            # Get raw results from quantum node
            raw_results = self.quantum_node.get_results(job_id)
            
            # Get job metadata
            job = self.quantum_node.get_job(job_id)
            
            # Apply error mitigation
            mitigated_results = self.error_mitigator.apply_all_mitigations(
                circuit=job.circuit,
                results=raw_results.get('counts', {}),
                backend_name=job.backend
            )
            
            # Record metrics
            self.monitoring.increment_counter('jobs_completed', labels={'backend': job.backend})
            
            self.monitoring.end_span(span_id, status="success")
            
            return {
                'job_id': job_id,
                'results': mitigated_results,
                'metadata': {
                    'backend': job.backend,
                    'error_mitigation_applied': True,
                }
            }
            
        except Exception as e:
            self.monitoring.increment_counter('jobs_failed')
            self.monitoring.end_span(span_id, status="error")
            raise
    
    def get_dashboard_data(self) -> dict:
        """Get real-time dashboard data."""
        return {
            'monitoring': self.monitoring.get_dashboard_data(),
            'security': self.security.get_security_stats(),
            'sovereignty': self.sovereignty.get_sovereignty_stats(),
            'error_mitigation': self.error_mitigator.get_mitigation_stats(),
        }


# Example usage
if __name__ == '__main__':
    # Initialize production node
    node = ProductionKinichNode('config.yaml')
    
    # Create user
    user = node.security.create_user(
        username='alice',
        role='SUBMITTER',
        blockchain_address='0x1234...'
    )
    
    # Generate API key
    api_key = node.security.generate_api_key(user.user_id)
    print(f"API Key: {api_key}")
    
    # Example circuit (placeholder)
    class MockCircuit:
        num_qubits = 5
        def size(self): return 10
        def depth(self): return 3
        def count_ops(self): return {'h': 5, 'cx': 5}
    
    circuit = MockCircuit()
    
    # Submit job
    try:
        job_id = node.submit_job(
            user_id=user.user_id,
            api_key=api_key,
            circuit=circuit,
            backend='spinq_local',
            data_classification=DataClassification.INTERNAL,
            shots=1024
        )
        print(f"Job submitted: {job_id}")
        
        # Get dashboard data
        dashboard = node.get_dashboard_data()
        print(f"Dashboard: {dashboard}")
        
    except Exception as e:
        print(f"Error: {e}")
