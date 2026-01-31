"""
Data Sovereignty Module for Kinich

Ensures Belize maintains full control over quantum computing data and operations.
Implements data residency, compliance, and national security controls.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"  # National security
    SOVEREIGN = "sovereign"  # Must stay in Belize


class ComplianceRegime(Enum):
    """Compliance regimes."""
    BELIZE_DATA_PROTECTION = "belize_dpa"
    GDPR = "gdpr"
    CCPA = "ccpa"
    CARICOM = "caricom"  # Caribbean regional


@dataclass
class DataResidencyRule:
    """Rule for data residency."""
    classification: DataClassification
    allowed_regions: Set[str]
    prohibited_regions: Set[str]
    requires_encryption: bool
    max_retention_days: Optional[int] = None


@dataclass
class ComplianceRecord:
    """Record of compliance action."""
    timestamp: datetime
    action: str
    data_classification: DataClassification
    user_id: str
    compliant: bool
    details: Dict[str, Any]


class SovereigntyManager:
    """
    Manages data sovereignty and compliance for Kinich.
    
    Ensures Belize maintains control over quantum computing data:
    - Data residency enforcement
    - Cross-border controls
    - Compliance tracking
    - National security filters
    - Government override mechanisms
    """
    
    def __init__(self):
        """Initialize sovereignty manager."""
        
        # Define Belize's data residency rules
        self._residency_rules = {
            DataClassification.PUBLIC: DataResidencyRule(
                classification=DataClassification.PUBLIC,
                allowed_regions={'*'},  # Allowed anywhere
                prohibited_regions=set(),
                requires_encryption=False,
            ),
            DataClassification.INTERNAL: DataResidencyRule(
                classification=DataClassification.INTERNAL,
                allowed_regions={'americas', 'caricom'},
                prohibited_regions={'cn', 'ru'},  # Example restrictions
                requires_encryption=True,
            ),
            DataClassification.CONFIDENTIAL: DataResidencyRule(
                classification=DataClassification.CONFIDENTIAL,
                allowed_regions={'belize', 'caricom'},
                prohibited_regions=set(),
                requires_encryption=True,
                max_retention_days=365,
            ),
            DataClassification.RESTRICTED: DataResidencyRule(
                classification=DataClassification.RESTRICTED,
                allowed_regions={'belize'},
                prohibited_regions=set(),
                requires_encryption=True,
                max_retention_days=180,
            ),
            DataClassification.SOVEREIGN: DataResidencyRule(
                classification=DataClassification.SOVEREIGN,
                allowed_regions={'belize'},
                prohibited_regions=set(),
                requires_encryption=True,
                max_retention_days=90,
            ),
        }
        
        # Backend region mapping
        self._backend_regions = {
            'azure_ionq': 'us',
            'azure_quantinuum': 'us',
            'ibm_quantum': 'us',
            'spinq_local': 'belize',  # Local hardware
        }
        
        # Compliance tracking
        self._compliance_records: List[ComplianceRecord] = []
        
        # National security filters (job types requiring government approval)
        self._restricted_job_types = {
            'cryptography_shor',
            'cryptography_grover',
            'national_security',
        }
        
        # Emergency controls
        self._emergency_mode = False
        self._national_override_active = False
        
        logger.info("Initialized sovereignty manager")
    
    # ==================== DATA RESIDENCY ====================
    
    def check_data_residency(
        self,
        data_classification: DataClassification,
        backend_name: str,
        user_id: str
    ) -> bool:
        """
        Check if data can be processed on backend.
        
        Args:
            data_classification: Data classification level
            backend_name: Target backend
            user_id: User submitting job
        
        Returns:
            True if compliant, False otherwise
        """
        rule = self._residency_rules[data_classification]
        backend_region = self._backend_regions.get(backend_name, 'unknown')
        
        # Check if region is allowed
        if '*' not in rule.allowed_regions:
            if backend_region not in rule.allowed_regions:
                logger.warning(
                    f"Data residency violation: {data_classification.value} "
                    f"not allowed in {backend_region}"
                )
                self._record_compliance_action(
                    action="data_residency_check",
                    data_classification=data_classification,
                    user_id=user_id,
                    compliant=False,
                    details={
                        'backend': backend_name,
                        'region': backend_region,
                        'reason': 'region_not_allowed'
                    }
                )
                return False
        
        # Check if region is prohibited
        if backend_region in rule.prohibited_regions:
            logger.warning(
                f"Data residency violation: {backend_region} is prohibited "
                f"for {data_classification.value}"
            )
            self._record_compliance_action(
                action="data_residency_check",
                data_classification=data_classification,
                user_id=user_id,
                compliant=False,
                details={
                    'backend': backend_name,
                    'region': backend_region,
                    'reason': 'region_prohibited'
                }
            )
            return False
        
        # Check encryption requirement
        if rule.requires_encryption:
            # In production, verify data is encrypted
            pass
        
        self._record_compliance_action(
            action="data_residency_check",
            data_classification=data_classification,
            user_id=user_id,
            compliant=True,
            details={
                'backend': backend_name,
                'region': backend_region,
            }
        )
        
        return True
    
    def enforce_data_localization(
        self,
        job_data: Dict[str, Any],
        classification: DataClassification
    ) -> bool:
        """
        Ensure sensitive data stays in Belize.
        
        Args:
            job_data: Job data to check
            classification: Data classification
        
        Returns:
            True if compliant
        """
        rule = self._residency_rules[classification]
        
        # For SOVEREIGN/RESTRICTED data, only allow local backends
        if classification in [DataClassification.SOVEREIGN, DataClassification.RESTRICTED]:
            backend = job_data.get('backend', '')
            if 'spinq_local' not in backend:
                logger.error(
                    f"{classification.value} data must use local backend, "
                    f"got {backend}"
                )
                return False
        
        return True
    
    # ==================== COMPLIANCE TRACKING ====================
    
    def _record_compliance_action(
        self,
        action: str,
        data_classification: DataClassification,
        user_id: str,
        compliant: bool,
        details: Dict[str, Any]
    ):
        """Record compliance action."""
        record = ComplianceRecord(
            timestamp=datetime.utcnow(),
            action=action,
            data_classification=data_classification,
            user_id=user_id,
            compliant=compliant,
            details=details
        )
        self._compliance_records.append(record)
    
    def generate_compliance_report(
        self,
        regime: ComplianceRegime,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report for regulatory regime.
        
        Args:
            regime: Compliance regime
            start_date: Report start date
            end_date: Report end date
        
        Returns:
            Compliance report
        """
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()
        
        # Filter records by date range
        filtered = [
            r for r in self._compliance_records
            if start_date <= r.timestamp <= end_date
        ]
        
        total = len(filtered)
        compliant = sum(1 for r in filtered if r.compliant)
        violations = total - compliant
        
        report = {
            'regime': regime.value,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
            },
            'summary': {
                'total_actions': total,
                'compliant': compliant,
                'violations': violations,
                'compliance_rate': compliant / total if total > 0 else 1.0,
            },
            'violations_by_type': {},
            'data_classifications': {},
        }
        
        # Analyze violations
        for record in filtered:
            if not record.compliant:
                reason = record.details.get('reason', 'unknown')
                report['violations_by_type'][reason] = \
                    report['violations_by_type'].get(reason, 0) + 1
            
            # Track by classification
            classification = record.data_classification.value
            report['data_classifications'][classification] = \
                report['data_classifications'].get(classification, 0) + 1
        
        logger.info(
            f"Generated {regime.value} compliance report: "
            f"{compliant}/{total} actions compliant"
        )
        
        return report
    
    def check_right_to_deletion(
        self,
        user_id: str,
        data_type: str
    ) -> bool:
        """
        Check if user can request data deletion (GDPR/DPA compliance).
        
        Args:
            user_id: User requesting deletion
            data_type: Type of data to delete
        
        Returns:
            True if deletion allowed
        """
        # Check if data is subject to legal hold
        # Check if data has minimum retention requirement
        # Verify user identity
        
        # For now, allow deletion of most data types
        protected_types = ['audit_log', 'compliance_record', 'financial_transaction']
        
        if data_type in protected_types:
            logger.warning(f"Cannot delete {data_type} - legal retention required")
            return False
        
        return True
    
    # ==================== NATIONAL SECURITY ====================
    
    def check_national_security_clearance(
        self,
        job_type: str,
        user_id: str,
        user_role: str
    ) -> bool:
        """
        Check if job requires national security clearance.
        
        Args:
            job_type: Type of quantum job
            user_id: User submitting job
            user_role: User's role
        
        Returns:
            True if clearance granted
        """
        if job_type not in self._restricted_job_types:
            return True
        
        # Only GOVERNMENT role can run restricted jobs
        if user_role != 'GOVERNMENT':
            logger.warning(
                f"National security clearance denied: "
                f"{user_id} attempted {job_type} without clearance"
            )
            self._record_compliance_action(
                action="security_clearance_check",
                data_classification=DataClassification.RESTRICTED,
                user_id=user_id,
                compliant=False,
                details={
                    'job_type': job_type,
                    'user_role': user_role,
                    'reason': 'insufficient_clearance'
                }
            )
            return False
        
        return True
    
    def activate_national_override(self, authorized_by: str) -> bool:
        """
        Activate national override for emergency situations.
        
        Args:
            authorized_by: Government official authorizing override
        
        Returns:
            True if activated
        """
        logger.critical(f"NATIONAL OVERRIDE ACTIVATED by {authorized_by}")
        self._national_override_active = True
        
        self._record_compliance_action(
            action="national_override_activate",
            data_classification=DataClassification.SOVEREIGN,
            user_id=authorized_by,
            compliant=True,
            details={
                'timestamp': datetime.utcnow().isoformat(),
                'reason': 'national_security'
            }
        )
        
        return True
    
    def deactivate_national_override(self, authorized_by: str) -> bool:
        """Deactivate national override."""
        logger.info(f"National override deactivated by {authorized_by}")
        self._national_override_active = False
        return True
    
    def emergency_shutdown(self, reason: str, authorized_by: str):
        """
        Emergency shutdown of all quantum operations.
        
        Args:
            reason: Reason for shutdown
            authorized_by: Government official authorizing shutdown
        """
        logger.critical(
            f"EMERGENCY SHUTDOWN INITIATED\n"
            f"Reason: {reason}\n"
            f"Authorized by: {authorized_by}"
        )
        
        self._emergency_mode = True
        
        self._record_compliance_action(
            action="emergency_shutdown",
            data_classification=DataClassification.SOVEREIGN,
            user_id=authorized_by,
            compliant=True,
            details={
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat(),
            }
        )
        
        # In production, this would:
        # - Stop all running jobs
        # - Lock down all backends
        # - Secure all data
        # - Alert authorities
    
    # ==================== STATISTICS ====================
    
    def get_sovereignty_stats(self) -> Dict[str, Any]:
        """Get sovereignty statistics."""
        total_checks = len(self._compliance_records)
        violations = sum(1 for r in self._compliance_records if not r.compliant)
        
        return {
            'total_compliance_checks': total_checks,
            'violations': violations,
            'compliance_rate': (total_checks - violations) / total_checks if total_checks > 0 else 1.0,
            'emergency_mode': self._emergency_mode,
            'national_override_active': self._national_override_active,
            'backend_regions': self._backend_regions,
            'data_classifications': list(DataClassification.__members__.keys()),
        }
