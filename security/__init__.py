"""
Security Module for Kinich

Enterprise-grade security for sovereign quantum computing infrastructure.
Provides authentication, authorization, encryption, and audit logging.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import hashlib
import hmac
import secrets
import json
from enum import Enum
import jwt

logger = logging.getLogger(__name__)


class Role(Enum):
    """User roles for RBAC."""
    
    ADMIN = "admin"  # Full access
    OPERATOR = "operator"  # Node operation
    SUBMITTER = "submitter"  # Submit jobs
    VIEWER = "viewer"  # Read-only
    GOVERNMENT = "government"  # National override


class Permission(Enum):
    """Granular permissions."""
    
    # Job permissions
    SUBMIT_JOB = "submit_job"
    CANCEL_JOB = "cancel_job"
    VIEW_JOB = "view_job"
    VIEW_ALL_JOBS = "view_all_jobs"
    
    # Node permissions
    MANAGE_NODE = "manage_node"
    VIEW_NODE_STATUS = "view_node_status"
    SHUTDOWN_NODE = "shutdown_node"
    
    # Backend permissions
    CONFIGURE_BACKEND = "configure_backend"
    VIEW_BACKEND_STATS = "view_backend_stats"
    
    # Security permissions
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOG = "view_audit_log"
    EMERGENCY_OVERRIDE = "emergency_override"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.OPERATOR: {
        Permission.SUBMIT_JOB,
        Permission.CANCEL_JOB,
        Permission.VIEW_JOB,
        Permission.VIEW_ALL_JOBS,
        Permission.MANAGE_NODE,
        Permission.VIEW_NODE_STATUS,
        Permission.CONFIGURE_BACKEND,
        Permission.VIEW_BACKEND_STATS,
    },
    Role.SUBMITTER: {
        Permission.SUBMIT_JOB,
        Permission.VIEW_JOB,
        Permission.VIEW_NODE_STATUS,
        Permission.VIEW_BACKEND_STATS,
    },
    Role.VIEWER: {
        Permission.VIEW_JOB,
        Permission.VIEW_NODE_STATUS,
        Permission.VIEW_BACKEND_STATS,
    },
    Role.GOVERNMENT: {
        Permission.EMERGENCY_OVERRIDE,
        Permission.SHUTDOWN_NODE,
        Permission.VIEW_AUDIT_LOG,
        Permission.VIEW_ALL_JOBS,
    },
}


@dataclass
class User:
    """User identity."""
    
    user_id: str
    username: str
    role: Role
    blockchain_address: Optional[str] = None
    api_key: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_login: Optional[str] = None
    active: bool = True


@dataclass
class AuditLogEntry:
    """Tamper-proof audit log entry."""
    
    timestamp: str
    user_id: str
    action: str
    resource: str
    result: str  # success/failure
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    signature: Optional[str] = None  # HMAC signature


class SecurityManager:
    """
    Manages authentication, authorization, and audit logging
    for Kinich quantum infrastructure.
    """
    
    def __init__(
        self,
        jwt_secret: str,
        audit_log_key: str,
        enable_encryption: bool = True
    ):
        """
        Initialize security manager.
        
        Args:
            jwt_secret: Secret key for JWT tokens
            audit_log_key: Secret key for audit log signatures
            enable_encryption: Enable end-to-end encryption
        """
        self.jwt_secret = jwt_secret
        self.audit_log_key = audit_log_key
        self.enable_encryption = enable_encryption
        
        # User management
        self._users: Dict[str, User] = {}
        self._api_keys: Dict[str, str] = {}  # api_key -> user_id
        
        # Audit logging
        self._audit_log: List[AuditLogEntry] = []
        
        # Rate limiting
        self._rate_limits: Dict[str, List[datetime]] = {}
        
        logger.info("Initialized security manager")
    
    # ==================== AUTHENTICATION ====================
    
    def create_user(
        self,
        username: str,
        role: Role,
        blockchain_address: Optional[str] = None
    ) -> User:
        """Create new user."""
        user_id = hashlib.sha256(f"{username}{datetime.now()}".encode()).hexdigest()[:16]
        
        user = User(
            user_id=user_id,
            username=username,
            role=role,
            blockchain_address=blockchain_address
        )
        
        self._users[user_id] = user
        
        self.audit_log(
            user_id="system",
            action="CREATE_USER",
            resource=f"user:{user_id}",
            result="success",
            details={"username": username, "role": role.value}
        )
        
        logger.info(f"Created user: {username} (role: {role.value})")
        return user
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate API key for user."""
        if user_id not in self._users:
            raise ValueError(f"User {user_id} not found")
        
        api_key = f"kinich_{secrets.token_urlsafe(32)}"
        
        self._users[user_id].api_key = api_key
        self._api_keys[api_key] = user_id
        
        self.audit_log(
            user_id=user_id,
            action="GENERATE_API_KEY",
            resource=f"user:{user_id}",
            result="success",
            details={}
        )
        
        return api_key
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate user by API key."""
        user_id = self._api_keys.get(api_key)
        
        if user_id:
            user = self._users.get(user_id)
            
            if user and user.active:
                user.last_login = datetime.now().isoformat()
                return user
        
        return None
    
    def authenticate_jwt(self, token: str) -> Optional[User]:
        """Authenticate user by JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            user_id = payload.get("user_id")
            
            if user_id:
                user = self._users.get(user_id)
                
                if user and user.active:
                    user.last_login = datetime.now().isoformat()
                    return user
        
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
        
        return None
    
    def generate_jwt(self, user_id: str, expires_hours: int = 24) -> str:
        """Generate JWT token for user."""
        if user_id not in self._users:
            raise ValueError(f"User {user_id} not found")
        
        user = self._users[user_id]
        
        payload = {
            "user_id": user_id,
            "username": user.username,
            "role": user.role.value,
            "exp": datetime.utcnow() + timedelta(hours=expires_hours),
            "iat": datetime.utcnow(),
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
        self.audit_log(
            user_id=user_id,
            action="GENERATE_JWT",
            resource=f"user:{user_id}",
            result="success",
            details={"expires_hours": expires_hours}
        )
        
        return token
    
    # ==================== AUTHORIZATION ====================
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has permission."""
        role_perms = ROLE_PERMISSIONS.get(user.role, set())
        return permission in role_perms
    
    def authorize_action(
        self,
        user: User,
        action: Permission,
        resource: str
    ) -> bool:
        """Authorize user action on resource."""
        if not user.active:
            self.audit_log(
                user_id=user.user_id,
                action=action.value,
                resource=resource,
                result="failure",
                details={"reason": "user_inactive"}
            )
            return False
        
        if not self.has_permission(user, action):
            self.audit_log(
                user_id=user.user_id,
                action=action.value,
                resource=resource,
                result="failure",
                details={"reason": "insufficient_permissions"}
            )
            return False
        
        # Check rate limits
        if not self._check_rate_limit(user.user_id):
            self.audit_log(
                user_id=user.user_id,
                action=action.value,
                resource=resource,
                result="failure",
                details={"reason": "rate_limit_exceeded"}
            )
            return False
        
        self.audit_log(
            user_id=user.user_id,
            action=action.value,
            resource=resource,
            result="success",
            details={}
        )
        
        return True
    
    def _check_rate_limit(
        self,
        user_id: str,
        max_requests: int = 100,
        window_seconds: int = 60
    ) -> bool:
        """Check rate limiting."""
        now = datetime.now()
        
        if user_id not in self._rate_limits:
            self._rate_limits[user_id] = []
        
        # Remove old requests outside window
        cutoff = now - timedelta(seconds=window_seconds)
        self._rate_limits[user_id] = [
            ts for ts in self._rate_limits[user_id]
            if ts > cutoff
        ]
        
        # Check limit
        if len(self._rate_limits[user_id]) >= max_requests:
            return False
        
        # Add current request
        self._rate_limits[user_id].append(now)
        return True
    
    # ==================== ENCRYPTION ====================
    
    def encrypt_job_data(self, data: Dict[str, Any], user_key: str) -> str:
        """
        Encrypt job data.
        
        ⚠️ PRODUCTION WARNING:
        This is a PLACEHOLDER implementation using simple XOR.
        For production use, REPLACE with:
        - cryptography.fernet.Fernet (symmetric encryption)
        - AES-256-GCM via cryptography.hazmat
        - Or use PostQuantumCrypto for quantum-resistant encryption
        
        Args:
            data: Data to encrypt
            user_key: Encryption key
            
        Returns:
            Hex-encoded encrypted data
        """
        import warnings
        warnings.warn(
            "encrypt_job_data uses INSECURE XOR encryption. "
            "Replace with AES-256-GCM or Fernet for production!",
            category=SecurityWarning,
            stacklevel=2
        )
        
        json_data = json.dumps(data)
        
        # INSECURE: XOR encryption (example only)
        # TODO: Replace with:
        # from cryptography.fernet import Fernet
        # fernet = Fernet(user_key)
        # encrypted = fernet.encrypt(json_data.encode())
        # return encrypted.hex()
        
        key_bytes = user_key.encode()
        encrypted = bytes([
            json_data.encode()[i] ^ key_bytes[i % len(key_bytes)]
            for i in range(len(json_data))
        ])
        
        return encrypted.hex()
    
    def decrypt_job_data(self, encrypted: str, user_key: str) -> Dict[str, Any]:
        """
        Decrypt job data.
        
        ⚠️ PRODUCTION WARNING: Uses insecure XOR - see encrypt_job_data()
        """
        import warnings
        warnings.warn(
            "decrypt_job_data uses INSECURE XOR decryption. "
            "Replace with AES-256-GCM or Fernet for production!",
            category=SecurityWarning,
            stacklevel=2
        )
        
        encrypted_bytes = bytes.fromhex(encrypted)
        key_bytes = user_key.encode()
        
        decrypted = bytes([
            encrypted_bytes[i] ^ key_bytes[i % len(key_bytes)]
            for i in range(len(encrypted_bytes))
        ])
        
        return json.loads(decrypted.decode())
    
    # ==================== AUDIT LOGGING ====================
    
    def audit_log(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        details: Dict[str, Any],
        ip_address: Optional[str] = None
    ) -> None:
        """Create tamper-proof audit log entry."""
        entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            details=details,
            ip_address=ip_address
        )
        
        # Create HMAC signature for tamper-proofing
        message = f"{entry.timestamp}|{entry.user_id}|{entry.action}|{entry.resource}|{entry.result}"
        entry.signature = hmac.new(
            self.audit_log_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        self._audit_log.append(entry)
        
        logger.info(
            f"AUDIT: {entry.user_id} {entry.action} {entry.resource} -> {entry.result}"
        )
    
    def verify_audit_log_entry(self, entry: AuditLogEntry) -> bool:
        """Verify audit log entry hasn't been tampered with."""
        message = f"{entry.timestamp}|{entry.user_id}|{entry.action}|{entry.resource}|{entry.result}"
        expected_signature = hmac.new(
            self.audit_log_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return entry.signature == expected_signature
    
    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditLogEntry]:
        """Query audit log."""
        results = self._audit_log
        
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        
        if action:
            results = [e for e in results if e.action == action]
        
        if start_time:
            results = [
                e for e in results
                if datetime.fromisoformat(e.timestamp) >= start_time
            ]
        
        if end_time:
            results = [
                e for e in results
                if datetime.fromisoformat(e.timestamp) <= end_time
            ]
        
        return results
    
    # ==================== SECURITY POLICIES ====================
    
    def validate_circuit_complexity(
        self,
        circuit: Any,
        max_qubits: int = 50,
        max_gates: int = 10000,
        max_depth: int = 1000
    ) -> bool:
        """Validate circuit doesn't exceed complexity limits."""
        try:
            num_qubits = circuit.num_qubits if hasattr(circuit, 'num_qubits') else 0
            num_gates = circuit.size() if hasattr(circuit, 'size') else 0
            depth = circuit.depth() if hasattr(circuit, 'depth') else 0
            
            if num_qubits > max_qubits:
                logger.warning(f"Circuit exceeds max qubits: {num_qubits} > {max_qubits}")
                return False
            
            if num_gates > max_gates:
                logger.warning(f"Circuit exceeds max gates: {num_gates} > {max_gates}")
                return False
            
            if depth > max_depth:
                logger.warning(f"Circuit exceeds max depth: {depth} > {max_depth}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Circuit validation error: {e}")
            return False
    
    def detect_malicious_circuit(self, circuit: Any) -> bool:
        """Detect potentially malicious circuits."""
        # Check for resource exhaustion patterns
        # Check for infinite loops
        # Check for excessive memory usage
        
        # Placeholder - implement actual detection
        return False
    
    # ==================== STATISTICS ====================
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        return {
            'total_users': len(self._users),
            'active_users': sum(1 for u in self._users.values() if u.active),
            'total_audit_entries': len(self._audit_log),
            'users_by_role': {
                role.value: sum(1 for u in self._users.values() if u.role == role)
                for role in Role
            },
        }


# ==================== ZERO-KNOWLEDGE PROOFS ====================

from kinich.security.zk_proofs import (
    ZKProofGenerator,
    ZKProof,
    BatchProof,
    ZKPublicInputs,
    ZKPrivateInputs,
    ProofSystem,
    CircuitType,
)

# ==================== POST-QUANTUM CRYPTOGRAPHY ====================

from kinich.security.post_quantum_crypto import (
    PostQuantumCrypto,
    PQCKeyPair,
    PQCSignature,
    QuantumKey,
    SignatureScheme,
    KEMScheme,
)

__all__ = [
    # Core security classes
    'SecurityManager',
    'User',
    'Role',
    'Permission',
    'AuditLogEntry',
    
    # Zero-knowledge proofs
    'ZKProofGenerator',
    'ZKProof',
    'BatchProof',
    'ZKPublicInputs',
    'ZKPrivateInputs',
    'ProofSystem',
    'CircuitType',
    
    # Post-quantum cryptography
    'PostQuantumCrypto',
    'PQCKeyPair',
    'PQCSignature',
    'QuantumKey',
    'SignatureScheme',
    'KEMScheme',
]
