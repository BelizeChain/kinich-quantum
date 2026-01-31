"""
Production Quantum Node

Production-hardened quantum node with security, sovereignty, error mitigation,
monitoring, and configuration management fully integrated.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging
import uuid

# Core quantum components
from ..core import (
    QuantumJob,
    JobStatus,
    JobPriority,
    QuantumJobResult,
    JobScheduler,
    CircuitOptimizer,
    ResultAggregator,
)
from ..core.quantum_node import QuantumNode, NodeConfig, NodeStatus

# Production hardening components
from ..security import SecurityManager, Role, Permission
from ..error_mitigation import QuantumErrorMitigator, ErrorMitigationConfig
from ..sovereignty import SovereigntyManager, DataClassification
from ..monitoring import MonitoringManager, AlertSeverity
from ..config import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class ProductionNodeConfig(NodeConfig):
    """Extended configuration for production node."""
    
    # Security settings
    enable_authentication: bool = True
    jwt_secret: Optional[str] = None
    
    # Sovereignty settings
    enforce_data_residency: bool = True
    default_data_classification: str = "INTERNAL"
    
    # Error mitigation settings
    enable_error_mitigation: bool = True
    enable_readout_mitigation: bool = True
    enable_zero_noise_extrapolation: bool = False
    
    # Monitoring settings
    enable_distributed_tracing: bool = True
    enable_alerting: bool = True
    
    # Configuration file
    config_file_path: Optional[str] = None


class ProductionQuantumNode(QuantumNode):
    """
    Production-hardened Kinich Quantum Node.
    
    Integrates comprehensive production capabilities:
    - Security: Authentication, authorization, audit logging
    - Sovereignty: Data residency, compliance tracking
    - Error Mitigation: Readout correction, ZNE, symmetry checks
    - Monitoring: Distributed tracing, metrics, alerting
    - Configuration: Validated config management
    
    All job submissions go through security checks, sovereignty validation,
    and results are error-mitigated before being returned.
    """
    
    def __init__(
        self,
        config: Optional[ProductionNodeConfig] = None,
        adapter_registry=None,
        config_manager: Optional[ConfigManager] = None
    ):
        """
        Initialize production quantum node.
        
        Args:
            config: Production node configuration
            adapter_registry: Adapter registry for quantum backends
            config_manager: Configuration manager (optional)
        """
        # Initialize configuration first
        self.config_manager = config_manager or ConfigManager(
            config.config_file_path if config else None
        )
        
        # Apply config overrides
        if config is None:
            config = self._build_config_from_manager()
        
        # Initialize base quantum node
        super().__init__(config, adapter_registry)
        
        # Initialize production hardening components
        self._initialize_security()
        self._initialize_sovereignty()
        self._initialize_error_mitigation()
        self._initialize_monitoring()
        
        # Override status
        self.status = NodeStatus.INITIALIZING
        
        logger.info(
            f"Initialized PRODUCTION quantum node: {self.node_id} "
            f"with full security, sovereignty, and monitoring"
        )
    
    def _build_config_from_manager(self) -> ProductionNodeConfig:
        """Build config from configuration manager."""
        return ProductionNodeConfig(
            node_id=self.config_manager.get('quantum.node_id', f"kinich-{uuid.uuid4().hex[:8]}"),
            node_name=self.config_manager.get('quantum.node_name', 'Kinich Production Node'),
            max_concurrent_jobs=self.config_manager.get('quantum.max_concurrent_jobs', 10),
            max_queue_size=self.config_manager.get('quantum.max_queue_size', 100),
            enable_authentication=self.config_manager.get('security.enable_authentication', True),
            enforce_data_residency=self.config_manager.get('sovereignty.enforce_data_residency', True),
            enable_error_mitigation=self.config_manager.get('error_mitigation.enable_readout_mitigation', True),
            enable_distributed_tracing=self.config_manager.get('monitoring.enable_tracing', True),
        )
    
    def _initialize_security(self):
        """Initialize security module."""
        jwt_secret = self.config_manager.get_secret('jwt_secret') or \
                     self.config_manager.get('security.jwt_secret', 'CHANGE_IN_PRODUCTION')
        
        audit_log_key = self.config_manager.get_secret('audit_log_key') or 'audit_key'
        
        self.security_manager = SecurityManager(
            jwt_secret=jwt_secret,
            audit_log_key=audit_log_key
        )
        
        # Set rate limit from config
        rate_limit_config = self.config_manager.get('security.rate_limit', {})
        if isinstance(rate_limit_config, dict):
            self.security_manager.rate_limit['requests'] = rate_limit_config.get('requests', 100)
            self.security_manager.rate_limit['window'] = rate_limit_config.get('window_seconds', 60)
        
        logger.info("✅ Security module initialized")
    
    def _initialize_sovereignty(self):
        """Initialize data sovereignty module."""
        self.sovereignty_manager = SovereigntyManager()
        
        logger.info("✅ Sovereignty module initialized")
    
    def _initialize_error_mitigation(self):
        """Initialize error mitigation module."""
        error_config = ErrorMitigationConfig(
            enable_readout_mitigation=self.config_manager.get(
                'error_mitigation.enable_readout_mitigation', True
            ),
            enable_zero_noise_extrapolation=self.config_manager.get(
                'error_mitigation.enable_zero_noise_extrapolation', False
            ),
            enable_dynamic_decoupling=self.config_manager.get(
                'error_mitigation.enable_dynamic_decoupling', False
            ),
            max_acceptable_error=self.config_manager.get(
                'error_mitigation.max_acceptable_error', 0.1
            ),
        )
        
        self.error_mitigator = QuantumErrorMitigator(error_config)
        
        logger.info("✅ Error mitigation module initialized")
    
    def _initialize_monitoring(self):
        """Initialize monitoring module."""
        self.monitoring = MonitoringManager()
        
        # Register alert callback
        self.monitoring.register_alert_callback(self._handle_alert)
        
        # Set up default alert rules from config
        if self.config_manager.get('monitoring.enable_alerts', True):
            self._setup_default_alerts()
        
        logger.info("✅ Monitoring module initialized")
    
    def _setup_default_alerts(self):
        """Set up default alert rules."""
        # High failure rate alert
        self.monitoring.add_alert_rule(
            'high_job_failure_rate',
            metric_name='jobs_failed',
            threshold=5,
            window_seconds=60,
            severity=AlertSeverity.ERROR,
            message='High job failure rate detected (>5 failures/min)'
        )
        
        # Queue overflow alert
        self.monitoring.add_alert_rule(
            'queue_near_capacity',
            metric_name='queue_size',
            threshold=self.config.max_queue_size * 0.9,
            severity=AlertSeverity.WARNING,
            message='Job queue near capacity'
        )
    
    def _handle_alert(self, alert):
        """Handle monitoring alerts."""
        logger.warning(
            f"🚨 ALERT [{alert.severity.value.upper()}]: {alert.message} "
            f"(metric={alert.metric_name}, value={alert.current_value})"
        )
        
        # In production, this could:
        # - Send notifications (email, Slack, PagerDuty)
        # - Trigger auto-scaling
        # - Activate emergency protocols
    
    async def start(self) -> None:
        """Start production quantum node with full monitoring."""
        span_id = self.monitoring.start_span(
            "node_startup",
            attributes={'node_id': self.node_id}
        )
        
        try:
            logger.info(f"Starting PRODUCTION quantum node {self.node_id}...")
            
            # Call parent start
            await super().start()
            
            # Record startup metric
            self.monitoring.increment_counter('node_started')
            self.monitoring.set_gauge('node_status', 1.0)  # 1.0 = ready
            
            self.monitoring.end_span(span_id, status="success")
            
            logger.info(
                f"🚀 PRODUCTION quantum node {self.node_id} is LIVE with "
                f"security, sovereignty, and monitoring active!"
            )
            
        except Exception as e:
            self.monitoring.end_span(span_id, status="error")
            logger.error(f"Failed to start production node: {e}")
            raise
    
    def submit_job_authenticated(
        self,
        job: QuantumJob,
        user_id: str,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        data_classification: DataClassification = DataClassification.INTERNAL
    ) -> bool:
        """
        Submit job with full authentication, authorization, and sovereignty checks.
        
        Args:
            job: Quantum job to execute
            user_id: User ID submitting job
            api_key: API key for authentication
            jwt_token: JWT token for authentication
            data_classification: Data classification level
        
        Returns:
            True if accepted, False if rejected
        """
        span_id = self.monitoring.start_span(
            "submit_job_authenticated",
            attributes={
                'job_id': job.get_job_id(),
                'user_id': user_id,
                'classification': data_classification.value
            }
        )
        
        try:
            # 1. AUTHENTICATE
            user = None
            if api_key:
                user = self.security_manager.authenticate_api_key(api_key)
            elif jwt_token:
                payload = self.security_manager.authenticate_jwt(jwt_token)
                if payload:
                    user = self.security_manager._users.get(payload.get('user_id'))
            
            if not user:
                logger.warning(f"Authentication failed for user {user_id}")
                self.monitoring.increment_counter('auth_failures')
                self.monitoring.end_span(span_id, status="error")
                return False
            
            self.monitoring.add_span_event(span_id, "authenticated", {'user': user.username})
            
            # 2. AUTHORIZE
            authorized = self.security_manager.authorize_action(
                user=user,
                permission=Permission.SUBMIT_JOB,
                resource=f"job:{job.get_job_id()}"
            )
            
            if not authorized:
                logger.warning(f"Authorization failed for user {user_id}")
                self.monitoring.increment_counter('authz_failures')
                self.monitoring.end_span(span_id, status="error")
                return False
            
            self.monitoring.add_span_event(span_id, "authorized")
            
            # 3. CHECK DATA SOVEREIGNTY
            backend = getattr(job, 'backend_preference', ['spinq_local'])[0]
            
            if not self.sovereignty_manager.check_data_residency(
                data_classification=data_classification,
                backend_name=backend,
                user_id=user_id
            ):
                logger.warning(
                    f"Data sovereignty violation: {data_classification.value} "
                    f"cannot run on {backend}"
                )
                self.monitoring.increment_counter('sovereignty_violations')
                self.monitoring.end_span(span_id, status="error")
                return False
            
            self.monitoring.add_span_event(span_id, "sovereignty_check_passed")
            
            # 4. VALIDATE CIRCUIT SECURITY
            if hasattr(job, 'circuit') and job.circuit:
                if not self.security_manager.validate_circuit_complexity(job.circuit):
                    logger.warning(f"Circuit complexity validation failed for job {job.get_job_id()}")
                    self.monitoring.increment_counter('circuit_validation_failures')
                    self.monitoring.end_span(span_id, status="error")
                    return False
            
            self.monitoring.add_span_event(span_id, "circuit_validated")
            
            # 5. CHECK ERROR BUDGET
            if hasattr(job, 'circuit') and job.circuit:
                if not self.error_mitigator.check_error_budget(job.circuit, backend):
                    logger.warning(f"Job {job.get_job_id()} exceeds error budget")
                    self.monitoring.add_span_event(
                        span_id,
                        "error_budget_warning",
                        {'message': 'High error rate expected'}
                    )
            
            # 6. SUBMIT TO BASE QUEUE
            success = self.submit_job(job)
            
            if success:
                self.monitoring.increment_counter('jobs_submitted_authenticated')
                self.monitoring.end_span(span_id, status="success")
                logger.info(
                    f"✅ Job {job.get_job_id()} submitted by {user.username} "
                    f"(classification: {data_classification.value})"
                )
            else:
                self.monitoring.increment_counter('job_submission_failures')
                self.monitoring.end_span(span_id, status="error")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in authenticated job submission: {e}")
            self.monitoring.increment_counter('job_submission_errors')
            self.monitoring.end_span(span_id, status="error")
            return False
    
    async def _execute_job(self, job: QuantumJob) -> None:
        """Execute job with error mitigation and full monitoring."""
        job_id = job.get_job_id()
        
        span_id = self.monitoring.start_span(
            "execute_job",
            attributes={
                'job_id': job_id,
                'job_type': job.job_type.name if hasattr(job, 'job_type') else 'unknown'
            }
        )
        
        logger.info(f"Executing job {job_id} with error mitigation")
        
        # Mark as active
        self._active_jobs[job_id] = job
        job.mark_started()
        
        try:
            # Build quantum circuit
            self.monitoring.add_span_event(span_id, "building_circuit")
            circuit = await self._build_circuit_for_job(job)
            
            if circuit is None:
                raise RuntimeError("Failed to build circuit")
            
            # Record circuit metrics
            if hasattr(circuit, 'num_qubits'):
                self.monitoring.observe_histogram('circuit_qubits', circuit.num_qubits)
            if hasattr(circuit, 'depth'):
                self.monitoring.observe_histogram('circuit_depth', circuit.depth())
            
            # Execute on backend
            self.monitoring.add_span_event(span_id, "executing_on_backend")
            raw_result = await self._execute_circuit(circuit, job)
            
            # Apply error mitigation
            self.monitoring.add_span_event(span_id, "applying_error_mitigation")
            backend_used = raw_result.backend_used
            
            if hasattr(raw_result, 'counts') and raw_result.counts:
                mitigated_counts = self.error_mitigator.apply_all_mitigations(
                    circuit=circuit,
                    results=raw_result.counts,
                    backend_name=backend_used
                )
                
                # Update result with mitigated counts
                raw_result.counts = mitigated_counts
                self.monitoring.increment_counter('error_mitigation_applied')
            
            # Mark as completed
            job.mark_completed(raw_result)
            self._completed_jobs[job_id] = raw_result
            self._total_jobs_completed += 1
            
            # Update metrics
            exec_time = job.get_execution_time()
            if exec_time:
                self._total_execution_time += exec_time
                self.monitoring.observe_histogram('job_duration_ms', exec_time * 1000)
            
            # Submit to blockchain
            if self.config.submit_results_to_chain:
                await self._submit_result_to_chain(job, raw_result)
            
            # Trigger completion handlers
            for handler in self._job_completed_handlers:
                try:
                    handler(job, raw_result)
                except Exception as e:
                    logger.error(f"Job completed handler failed: {e}")
            
            self.monitoring.increment_counter('jobs_completed')
            self.monitoring.end_span(span_id, status="success")
            
            logger.info(
                f"✅ Job {job_id} completed successfully with error mitigation "
                f"(backend: {backend_used})"
            )
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            
            job.mark_failed(str(e))
            self._total_jobs_failed += 1
            
            self.monitoring.increment_counter('jobs_failed')
            self.monitoring.end_span(span_id, status="error")
        
        finally:
            # Remove from active jobs
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]
            
            # Update queue size metric
            self.monitoring.set_gauge('queue_size', len(self._job_queue))
            self.monitoring.set_gauge('active_jobs', len(self._active_jobs))
    
    def get_production_status(self) -> Dict[str, Any]:
        """Get comprehensive production status."""
        base_status = self.get_status()
        
        # Add production component statistics
        base_status['production'] = {
            'security': self.security_manager.get_security_stats(),
            'sovereignty': self.sovereignty_manager.get_sovereignty_stats(),
            'error_mitigation': self.error_mitigator.get_mitigation_stats(),
            'monitoring': self.monitoring.get_monitoring_stats(),
        }
        
        # Add SLA compliance
        base_status['sla_compliance'] = self.monitoring.check_sla_compliance()
        
        # Add active alerts
        base_status['active_alerts'] = len(self.monitoring.get_active_alerts())
        
        return base_status
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data."""
        return {
            'node': {
                'id': self.node_id,
                'name': self.config.node_name,
                'status': self.status.value,
            },
            'jobs': {
                'queued': len(self._job_queue),
                'active': len(self._active_jobs),
                'completed': self._total_jobs_completed,
                'failed': self._total_jobs_failed,
                'success_rate': (
                    self._total_jobs_completed / self._total_jobs_received
                    if self._total_jobs_received > 0 else 0.0
                ),
            },
            'monitoring': self.monitoring.get_dashboard_data(),
            'security': {
                'total_users': self.security_manager.get_security_stats()['total_users'],
                'audit_entries': self.security_manager.get_security_stats()['total_audit_entries'],
            },
            'sovereignty': {
                'compliance_rate': self.sovereignty_manager.get_sovereignty_stats()['compliance_rate'],
                'emergency_mode': self.sovereignty_manager.get_sovereignty_stats()['emergency_mode'],
            },
            'sla_compliance': self.monitoring.check_sla_compliance(),
            'alerts': {
                'active': len(self.monitoring.get_active_alerts()),
                'total': self.monitoring.get_monitoring_stats()['total_alerts'],
            }
        }
    
    async def stop(self) -> None:
        """Stop production quantum node with graceful shutdown."""
        span_id = self.monitoring.start_span("node_shutdown")
        
        logger.info(f"Stopping PRODUCTION quantum node {self.node_id}...")
        
        try:
            # Call parent stop
            await super().stop()
            
            # Update metrics
            self.monitoring.set_gauge('node_status', 0.0)  # 0.0 = offline
            self.monitoring.increment_counter('node_stopped')
            
            self.monitoring.end_span(span_id, status="success")
            
            logger.info(f"✅ Production quantum node {self.node_id} stopped cleanly")
            
        except Exception as e:
            self.monitoring.end_span(span_id, status="error")
            logger.error(f"Error during shutdown: {e}")
            raise
