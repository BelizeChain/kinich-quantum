"""
Monitoring & Observability for Kinich

Distributed tracing, metrics, and alerting for quantum operations.
Provides visibility into quantum job lifecycle and system health.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Span:
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "in_progress"  # in_progress, success, error
    
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


@dataclass
class Metric:
    """Metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_name: Optional[str] = None
    threshold: Optional[float] = None
    current_value: Optional[float] = None
    resolved: bool = False


class MonitoringManager:
    """
    Comprehensive monitoring and observability for Kinich.
    
    Provides:
    - Distributed tracing (OpenTelemetry-compatible)
    - Prometheus-style metrics
    - Alerting system
    - Real-time dashboards
    - SLA tracking
    """
    
    def __init__(self):
        """Initialize monitoring manager."""
        
        # Tracing
        self._active_spans: Dict[str, Span] = {}
        self._completed_traces: List[Span] = []
        
        # Metrics
        self._metrics: Dict[str, List[Metric]] = {}
        self._metric_callbacks: Dict[str, Callable] = {}
        
        # Alerts
        self._alerts: List[Alert] = []
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._alert_callbacks: List[Callable] = []
        
        # SLA tracking
        self._sla_targets = {
            'job_success_rate': 0.95,  # 95%
            'job_latency_p99': 60000,  # 60s in ms
            'system_uptime': 0.999,  # 99.9%
        }
        
        self._initialize_default_metrics()
        self._initialize_default_alerts()
        
        logger.info("Initialized monitoring manager")
    
    # ==================== DISTRIBUTED TRACING ====================
    
    def start_span(
        self,
        operation: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new tracing span.
        
        Args:
            operation: Operation name
            trace_id: Trace ID (generated if None)
            parent_span_id: Parent span ID for nested spans
            attributes: Initial attributes
        
        Returns:
            Span ID
        """
        import uuid
        
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
            attributes=attributes or {}
        )
        
        self._active_spans[span_id] = span
        
        logger.debug(f"Started span: {operation} (trace={trace_id[:8]}, span={span_id[:8]})")
        
        return span_id
    
    def add_span_event(
        self,
        span_id: str,
        event_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Add event to span."""
        if span_id not in self._active_spans:
            logger.warning(f"Span {span_id} not found")
            return
        
        span = self._active_spans[span_id]
        span.events.append({
            'name': event_name,
            'timestamp': time.time(),
            'attributes': attributes or {}
        })
    
    def set_span_attribute(self, span_id: str, key: str, value: Any):
        """Set span attribute."""
        if span_id not in self._active_spans:
            logger.warning(f"Span {span_id} not found")
            return
        
        self._active_spans[span_id].attributes[key] = value
    
    def end_span(self, span_id: str, status: str = "success"):
        """
        End a span.
        
        Args:
            span_id: Span ID
            status: Span status (success, error)
        """
        if span_id not in self._active_spans:
            logger.warning(f"Span {span_id} not found")
            return
        
        span = self._active_spans[span_id]
        span.end_time = time.time()
        span.status = status
        
        # Move to completed traces
        self._completed_traces.append(span)
        del self._active_spans[span_id]
        
        duration = span.duration_ms()
        logger.debug(f"Ended span: {span.operation} ({duration:.2f}ms, status={status})")
        
        # Record latency metric
        if duration is not None:
            self.record_metric(
                f"span_duration_{span.operation}",
                duration,
                MetricType.HISTOGRAM,
                labels={'status': status}
            )
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return [s for s in self._completed_traces if s.trace_id == trace_id]
    
    # ==================== METRICS ====================
    
    def _initialize_default_metrics(self):
        """Initialize default metrics."""
        self._metrics = {
            'jobs_submitted': [],
            'jobs_completed': [],
            'jobs_failed': [],
            'job_duration_ms': [],
            'circuit_depth': [],
            'qubits_used': [],
            'backend_availability': [],
            'error_mitigation_applied': [],
        }
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Metric labels
        """
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            labels=labels or {}
        )
        
        if name not in self._metrics:
            self._metrics[name] = []
        
        self._metrics[name].append(metric)
        
        # Keep last 10000 data points per metric
        if len(self._metrics[name]) > 10000:
            self._metrics[name] = self._metrics[name][-10000:]
        
        # Check alert rules
        self._check_alert_rules(name, value)
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.record_metric(name, 1.0, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, labels)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a histogram value."""
        self.record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def get_metric_summary(
        self,
        name: str,
        time_window_seconds: int = 300
    ) -> Dict[str, float]:
        """
        Get summary statistics for metric.
        
        Args:
            name: Metric name
            time_window_seconds: Time window for summary
        
        Returns:
            Summary statistics (count, sum, avg, min, max, p95, p99)
        """
        if name not in self._metrics:
            return {}
        
        cutoff = time.time() - time_window_seconds
        recent = [m for m in self._metrics[name] if m.timestamp >= cutoff]
        
        if not recent:
            return {}
        
        values = [m.value for m in recent]
        sorted_values = sorted(values)
        
        return {
            'count': len(values),
            'sum': sum(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'p50': sorted_values[len(sorted_values) // 2],
            'p95': sorted_values[int(len(sorted_values) * 0.95)],
            'p99': sorted_values[int(len(sorted_values) * 0.99)],
        }
    
    # ==================== ALERTING ====================
    
    def _initialize_default_alerts(self):
        """Initialize default alert rules."""
        self.add_alert_rule(
            'high_job_failure_rate',
            metric_name='jobs_failed',
            threshold=5,
            window_seconds=60,
            severity=AlertSeverity.ERROR,
            message='High job failure rate detected'
        )
        
        self.add_alert_rule(
            'backend_unavailable',
            metric_name='backend_availability',
            threshold=0.5,
            comparison='<',
            severity=AlertSeverity.CRITICAL,
            message='Backend availability below 50%'
        )
    
    def add_alert_rule(
        self,
        rule_name: str,
        metric_name: str,
        threshold: float,
        severity: AlertSeverity,
        message: str,
        window_seconds: int = 60,
        comparison: str = '>'
    ):
        """
        Add an alert rule.
        
        Args:
            rule_name: Unique rule name
            metric_name: Metric to monitor
            threshold: Alert threshold
            severity: Alert severity
            message: Alert message
            window_seconds: Evaluation window
            comparison: Comparison operator (>, <, ==)
        """
        self._alert_rules[rule_name] = {
            'metric_name': metric_name,
            'threshold': threshold,
            'severity': severity,
            'message': message,
            'window_seconds': window_seconds,
            'comparison': comparison,
        }
        
        logger.info(f"Added alert rule: {rule_name}")
    
    def _check_alert_rules(self, metric_name: str, current_value: float):
        """Check if any alert rules are triggered."""
        for rule_name, rule in self._alert_rules.items():
            if rule['metric_name'] != metric_name:
                continue
            
            threshold = rule['threshold']
            comparison = rule['comparison']
            
            # Evaluate condition
            triggered = False
            if comparison == '>' and current_value > threshold:
                triggered = True
            elif comparison == '<' and current_value < threshold:
                triggered = True
            elif comparison == '==' and current_value == threshold:
                triggered = True
            
            if triggered:
                self._trigger_alert(rule_name, rule, current_value)
    
    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], current_value: float):
        """Trigger an alert."""
        import uuid
        
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            severity=rule['severity'],
            message=rule['message'],
            timestamp=datetime.utcnow(),
            metric_name=rule['metric_name'],
            threshold=rule['threshold'],
            current_value=current_value,
        )
        
        self._alerts.append(alert)
        
        logger.warning(
            f"ALERT [{alert.severity.value.upper()}]: {alert.message} "
            f"({alert.metric_name}={current_value:.2f}, threshold={alert.threshold})"
        )
        
        # Call alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """Register callback for alerts."""
        self._alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts."""
        return [a for a in self._alerts if not a.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"Resolved alert: {alert_id}")
                break
    
    # ==================== SLA TRACKING ====================
    
    def check_sla_compliance(self) -> Dict[str, bool]:
        """Check SLA compliance for all targets."""
        compliance = {}
        
        # Job success rate
        jobs_completed = len([m for m in self._metrics.get('jobs_completed', [])])
        jobs_failed = len([m for m in self._metrics.get('jobs_failed', [])])
        total_jobs = jobs_completed + jobs_failed
        
        if total_jobs > 0:
            success_rate = jobs_completed / total_jobs
            compliance['job_success_rate'] = success_rate >= self._sla_targets['job_success_rate']
        
        # Job latency p99
        latency_summary = self.get_metric_summary('job_duration_ms', time_window_seconds=3600)
        if 'p99' in latency_summary:
            compliance['job_latency_p99'] = latency_summary['p99'] <= self._sla_targets['job_latency_p99']
        
        return compliance
    
    # ==================== STATISTICS & DASHBOARD ====================
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            'metrics': {
                name: self.get_metric_summary(name)
                for name in ['jobs_submitted', 'jobs_completed', 'jobs_failed', 'job_duration_ms']
            },
            'active_traces': len(self._active_spans),
            'completed_traces': len(self._completed_traces),
            'active_alerts': len(self.get_active_alerts()),
            'sla_compliance': self.check_sla_compliance(),
        }
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            'total_metrics_collected': sum(len(m) for m in self._metrics.values()),
            'metric_types': len(self._metrics),
            'active_spans': len(self._active_spans),
            'completed_traces': len(self._completed_traces),
            'total_alerts': len(self._alerts),
            'active_alerts': len(self.get_active_alerts()),
            'alert_rules': len(self._alert_rules),
        }
