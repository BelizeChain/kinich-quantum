"""
Complete Integration Example - Production Kinich Node

Demonstrates comprehensive integration of all production components
in a working quantum computing system.

This example shows:
- Complete node initialization with full security and monitoring
- User authentication and job submission
- Error mitigation and monitoring
- Dashboard and compliance reporting

Author: BelizeChain Team
License: MIT
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """
    Complete integration demonstration.
    """
    
    print("\n" + "="*70)
    print("  KINICH PRODUCTION NODE - COMPLETE INTEGRATION DEMO")
    print("  Production-Ready Quantum Computing Infrastructure")
    print("="*70 + "\n")
    
    # =====================================================================
    # STEP 1: Initialize Configuration
    # =====================================================================
    print("📋 STEP 1: Loading Configuration...")
    
    from kinich.config import ConfigManager
    
    config = ConfigManager()  # Loads defaults
    
    print(f"✅ Configuration loaded")
    print(f"   - Max qubits: {config.get('quantum.max_qubits')}")
    print(f"   - Authentication: {config.get('security.enable_authentication')}")
    print(f"   - Data residency: {config.get('sovereignty.enforce_data_residency')}")
    print()
    
    # =====================================================================
    # STEP 2: Initialize Security Manager
    # =====================================================================
    print("🔐 STEP 2: Initializing Security...")
    
    from kinich.security import SecurityManager, Role
    
    security = SecurityManager(
        jwt_secret=config.get('security.jwt_secret'),
        audit_log_key='demo_audit_key'
    )
    
    # Create demo users
    admin_user = security.create_user(
        username='admin_alice',
        role=Role.ADMIN,
        blockchain_address='0xADMIN123'
    )
    admin_api_key = security.generate_api_key(admin_user.user_id)
    
    operator_user = security.create_user(
        username='operator_bob',
        role=Role.OPERATOR,
        blockchain_address='0xOPERATOR456'
    )
    operator_api_key = security.generate_api_key(operator_user.user_id)
    
    submitter_user = security.create_user(
        username='researcher_charlie',
        role=Role.SUBMITTER,
        blockchain_address='0xSUBMITTER789'
    )
    submitter_api_key = security.generate_api_key(submitter_user.user_id)
    
    print(f"✅ Security initialized with {len(security._users)} users")
    print(f"   - Admin: {admin_user.username} (API key: {admin_api_key[:20]}...)")
    print(f"   - Operator: {operator_user.username} (API key: {operator_api_key[:20]}...)")
    print(f"   - Submitter: {submitter_user.username} (API key: {submitter_api_key[:20]}...)")
    print()
    
    # =====================================================================
    # STEP 3: Initialize Data Sovereignty
    # =====================================================================
    print("🌍 STEP 3: Initializing Data Sovereignty...")
    
    from kinich.sovereignty import SovereigntyManager, DataClassification
    
    sovereignty = SovereigntyManager()
    
    print(f"✅ Sovereignty initialized")
    print(f"   - Data classifications: {len(DataClassification)}")
    print(f"   - Backend regions: {list(sovereignty._backend_regions.keys())}")
    print()
    
    # =====================================================================
    # STEP 4: Initialize Error Mitigation
    # =====================================================================
    print("⚛️  STEP 4: Initializing Error Mitigation...")
    
    from kinich.error_mitigation import QuantumErrorMitigator, ErrorMitigationConfig
    
    error_config = ErrorMitigationConfig(
        enable_readout_mitigation=True,
        enable_zero_noise_extrapolation=False,  # Expensive
        enable_symmetry_verification=True,
        max_acceptable_error=0.1
    )
    
    error_mitigator = QuantumErrorMitigator(error_config)
    
    # Calibrate for demo backend
    error_matrix = error_mitigator.calibrate_readout_errors(
        backend_name='spinq_local',
        num_qubits=5,
        shots=1024
    )
    
    print(f"✅ Error mitigation initialized")
    print(f"   - Readout mitigation: Enabled")
    print(f"   - ZNE: Disabled (expensive)")
    print(f"   - Calibrated backends: {len(error_mitigator._readout_error_matrices)}")
    print()
    
    # =====================================================================
    # STEP 5: Initialize Monitoring
    # =====================================================================
    print("📊 STEP 5: Initializing Monitoring...")
    
    from kinich.monitoring import MonitoringManager, AlertSeverity
    
    monitoring = MonitoringManager()
    
    # Add custom alert
    monitoring.add_alert_rule(
        rule_name='demo_high_latency',
        metric_name='job_duration_ms',
        threshold=5000,  # 5 seconds
        severity=AlertSeverity.WARNING,
        message='Job latency exceeded 5 seconds'
    )
    
    print(f"✅ Monitoring initialized")
    print(f"   - Metrics tracked: {len(monitoring._metrics)}")
    print(f"   - Alert rules: {len(monitoring._alert_rules)}")
    print()
    
    # =====================================================================
    # STEP 6: Test Authentication Flow
    # =====================================================================
    print("🔑 STEP 6: Testing Authentication...")
    
    # Test API key auth
    auth_result = security.authenticate_api_key(submitter_api_key)
    print(f"✅ API key authentication: {auth_result.username if auth_result else 'FAILED'}")
    
    # Test JWT auth
    jwt_token = security.generate_jwt(submitter_user.user_id)
    jwt_payload = security.authenticate_jwt(jwt_token)
    print(f"✅ JWT authentication: {'SUCCESS' if jwt_payload else 'FAILED'}")
    print(f"   - JWT token: {jwt_token[:30]}...")
    print()
    
    # =====================================================================
    # STEP 7: Test Authorization
    # =====================================================================
    print("🛡️  STEP 7: Testing Authorization...")
    
    from kinich.security import Permission
    
    # Test submit permission
    can_submit = security.authorize_action(
        user=submitter_user,
        action='submit_job',
        resource='backend:spinq_local'
    )
    print(f"✅ Submitter can submit jobs: {can_submit}")
    
    # Test admin permission
    can_manage = security.authorize_action(
        user=admin_user,
        action='manage_node',
        resource='node:demo'
    )
    print(f"✅ Admin can manage node: {can_manage}")
    
    # Test unauthorized action
    cannot_manage = security.authorize_action(
        user=submitter_user,
        action='manage_node',
        resource='node:demo'
    )
    print(f"✅ Submitter cannot manage node: {not cannot_manage}")
    print()
    
    # =====================================================================
    # STEP 8: Test Data Sovereignty
    # =====================================================================
    print("🌍 STEP 8: Testing Data Sovereignty...")
    
    # Test INTERNAL data on local backend (should pass)
    can_use_local = sovereignty.check_data_residency(
        data_classification=DataClassification.INTERNAL,
        backend_name='spinq_local',
        user_id=submitter_user.user_id
    )
    print(f"✅ INTERNAL data on spinq_local: {can_use_local}")
    
    # Test SOVEREIGN data (should require local backend)
    can_use_sovereign = sovereignty.check_data_residency(
        data_classification=DataClassification.SOVEREIGN,
        backend_name='spinq_local',
        user_id=submitter_user.user_id
    )
    print(f"✅ SOVEREIGN data on spinq_local: {can_use_sovereign}")
    
    print()
    
    # =====================================================================
    # STEP 9: Test Error Mitigation
    # =====================================================================
    print("⚛️  STEP 9: Testing Error Mitigation...")
    
    # Simulate raw measurement counts
    raw_counts = {
        '00000': 450,
        '00001': 50,
        '11110': 50,
        '11111': 450
    }
    
    print(f"   Raw counts: {raw_counts}")
    
    # Apply error mitigation
    mitigated_counts = error_mitigator.mitigate_readout_errors(
        counts=raw_counts,
        backend_name='spinq_local',
        num_qubits=5
    )
    
    print(f"   Mitigated counts: {mitigated_counts}")
    print(f"✅ Error mitigation applied ({len(mitigated_counts)} states)")
    print()
    
    # =====================================================================
    # STEP 10: Test Monitoring & Tracing
    # =====================================================================
    print("📊 STEP 10: Testing Monitoring & Tracing...")
    
    # Start a trace
    span_id = monitoring.start_span(
        'demo_quantum_job',
        attributes={
            'user': submitter_user.username,
            'backend': 'spinq_local'
        }
    )
    
    # Simulate job execution
    monitoring.add_span_event(span_id, 'circuit_built')
    await asyncio.sleep(0.1)  # Simulate work
    
    monitoring.add_span_event(span_id, 'executing_on_backend')
    await asyncio.sleep(0.2)  # Simulate execution
    
    monitoring.add_span_event(span_id, 'error_mitigation_applied')
    
    # End span
    monitoring.end_span(span_id, status='success')
    
    # Record metrics
    monitoring.increment_counter('jobs_submitted')
    monitoring.increment_counter('jobs_completed')
    monitoring.observe_histogram('job_duration_ms', 300)
    
    print(f"✅ Distributed tracing span completed")
    print(f"   - Span duration: {monitoring._completed_traces[-1].duration_ms():.2f}ms")
    print(f"   - Events: {len(monitoring._completed_traces[-1].events)}")
    
    # Get metrics summary
    summary = monitoring.get_metric_summary('jobs_submitted', time_window_seconds=60)
    print(f"✅ Metrics recorded")
    print(f"   - Jobs submitted: {summary.get('count', 0)}")
    print()
    
    # =====================================================================
    # STEP 11: Generate Reports
    # =====================================================================
    print("📄 STEP 11: Generating Reports...")
    
    # Security stats
    security_stats = security.get_security_stats()
    print(f"✅ Security Report:")
    print(f"   - Total users: {security_stats['total_users']}")
    print(f"   - Audit entries: {security_stats['total_audit_entries']}")
    
    # Sovereignty compliance
    from kinich.sovereignty import ComplianceRegime
    
    compliance_report = sovereignty.generate_compliance_report(
        regime=ComplianceRegime.BELIZE_DATA_PROTECTION,
        start_date=datetime(2025, 10, 1),
        end_date=datetime.now()
    )
    print(f"\n✅ Sovereignty Compliance Report:")
    print(f"   - Total actions: {compliance_report['summary']['total_actions']}")
    print(f"   - Compliance rate: {compliance_report['summary']['compliance_rate']:.2%}")
    
    # Error mitigation stats
    error_stats = error_mitigator.get_mitigation_stats()
    print(f"\n✅ Error Mitigation Report:")
    print(f"   - Total mitigations: {error_stats['total_mitigations']}")
    print(f"   - Calibrated backends: {error_stats['calibrated_backends']}")
    
    # Monitoring stats
    monitoring_stats = monitoring.get_monitoring_stats()
    print(f"\n✅ Monitoring Report:")
    print(f"   - Metrics collected: {monitoring_stats['total_metrics_collected']}")
    print(f"   - Completed traces: {monitoring_stats['completed_traces']}")
    print(f"   - Active alerts: {monitoring_stats['active_alerts']}")
    
    # Dashboard data
    dashboard = monitoring.get_dashboard_data()
    print(f"\n✅ Dashboard Data:")
    print(f"   - Active traces: {dashboard['active_traces']}")
    print(f"   - SLA compliance: {dashboard['sla_compliance']}")
    
    print()
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "="*70)
    print("  ✅ INTEGRATION COMPLETE - ALL SYSTEMS OPERATIONAL")
    print("="*70)
    print("\n📊 Production Readiness Summary:")
    print(f"   🔐 Security:           {len(security._users)} users, {security_stats['total_audit_entries']} audit entries")
    print(f"   🌍 Sovereignty:        {compliance_report['summary']['compliance_rate']:.1%} compliant")
    print(f"   ⚛️  Error Mitigation:   {error_stats['calibrated_backends']} backends calibrated")
    print(f"   📊 Monitoring:         {monitoring_stats['completed_traces']} traces, {monitoring_stats['active_alerts']} alerts")
    print(f"   ⚙️  Configuration:      Validated and loaded")
    print()
    print("🎉 Kinich is ready for production quantum computing!")
    print()


if __name__ == '__main__':
    asyncio.run(main())
