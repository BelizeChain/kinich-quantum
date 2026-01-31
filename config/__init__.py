"""
Configuration Management for Kinich

Secure, validated configuration for quantum computing infrastructure.
Supports YAML/TOML files, environment variables, and secure secrets.

Author: BelizeChain Team
License: MIT
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
import os
import json

logger = logging.getLogger(__name__)


@dataclass
class ConfigValidationError(Exception):
    """Configuration validation error."""
    field: str
    message: str


class ConfigManager:
    """
    Configuration management for Kinich.
    
    Provides:
    - YAML/TOML config file loading
    - Environment variable override
    - Secure secrets management
    - Schema validation
    - Hot reload capability
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to config file (YAML or TOML)
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._secrets: Dict[str, str] = {}
        self._watchers: List[Callable] = []
        
        # Define required configuration schema
        self._schema = self._define_schema()
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        else:
            self.load_default_config()
        
        logger.info("Initialized configuration manager")
    
    # ==================== SCHEMA DEFINITION ====================
    
    def _define_schema(self) -> Dict[str, Any]:
        """Define configuration schema."""
        return {
            'quantum': {
                'required': True,
                'type': dict,
                'schema': {
                    'default_backend': {
                        'required': True,
                        'type': str,
                        'allowed': ['azure_ionq', 'ibm_quantum', 'spinq_local']
                    },
                    'max_qubits': {
                        'required': True,
                        'type': int,
                        'min': 1,
                        'max': 100
                    },
                    'default_shots': {
                        'required': True,
                        'type': int,
                        'min': 100,
                        'max': 100000
                    },
                }
            },
            'security': {
                'required': True,
                'type': dict,
                'schema': {
                    'enable_authentication': {
                        'required': True,
                        'type': bool
                    },
                    'jwt_secret': {
                        'required': True,
                        'type': str,
                        'min_length': 32
                    },
                    'rate_limit': {
                        'required': True,
                        'type': dict,
                        'schema': {
                            'requests': {'required': True, 'type': int, 'min': 1},
                            'window_seconds': {'required': True, 'type': int, 'min': 1}
                        }
                    }
                }
            },
            'sovereignty': {
                'required': True,
                'type': dict,
                'schema': {
                    'enforce_data_residency': {
                        'required': True,
                        'type': bool
                    },
                    'allowed_regions': {
                        'required': True,
                        'type': list
                    }
                }
            },
            'monitoring': {
                'required': False,
                'type': dict,
                'schema': {
                    'enable_tracing': {
                        'required': True,
                        'type': bool
                    },
                    'enable_metrics': {
                        'required': True,
                        'type': bool
                    }
                }
            }
        }
    
    # ==================== CONFIG LOADING ====================
    
    def load_default_config(self):
        """Load default configuration."""
        self._config = {
            'quantum': {
                'default_backend': 'spinq_local',
                'max_qubits': 50,
                'default_shots': 1024,
                'timeout_seconds': 300,
            },
            'security': {
                'enable_authentication': True,
                'jwt_secret': os.environ.get('KINICH_JWT_SECRET', 'CHANGE_ME_IN_PRODUCTION'),
                'rate_limit': {
                    'requests': 100,
                    'window_seconds': 60
                },
                'max_circuit_depth': 1000,
                'max_gates': 10000,
            },
            'sovereignty': {
                'enforce_data_residency': True,
                'allowed_regions': ['belize', 'caricom'],
                'enable_compliance_tracking': True,
            },
            'monitoring': {
                'enable_tracing': True,
                'enable_metrics': True,
                'enable_alerts': True,
            },
            'error_mitigation': {
                'enable_readout_mitigation': True,
                'enable_zero_noise_extrapolation': False,  # Expensive
                'enable_dynamic_decoupling': False,
            }
        }
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate
        self.validate_config()
    
    def load_config(self, config_path: str):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to config file (YAML or TOML)
        """
        path = Path(config_path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            self.load_default_config()
            return
        
        # Load based on file extension
        if path.suffix in ['.yaml', '.yml']:
            self._load_yaml(path)
        elif path.suffix == '.toml':
            self._load_toml(path)
        elif path.suffix == '.json':
            self._load_json(path)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate
        self.validate_config()
        
        logger.info(f"Loaded configuration from {config_path}")
    
    def _load_yaml(self, path: Path):
        """Load YAML configuration."""
        try:
            import yaml
            with open(path, 'r') as f:
                self._config = yaml.safe_load(f)
        except ImportError:
            logger.error("PyYAML not installed. Install with: pip install pyyaml")
            self.load_default_config()
        except Exception as e:
            logger.error(f"Failed to load YAML config: {e}")
            self.load_default_config()
    
    def _load_toml(self, path: Path):
        """Load TOML configuration."""
        try:
            import tomli
            with open(path, 'rb') as f:
                self._config = tomli.load(f)
        except ImportError:
            logger.error("tomli not installed. Install with: pip install tomli")
            self.load_default_config()
        except Exception as e:
            logger.error(f"Failed to load TOML config: {e}")
            self.load_default_config()
    
    def _load_json(self, path: Path):
        """Load JSON configuration."""
        try:
            with open(path, 'r') as f:
                self._config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON config: {e}")
            self.load_default_config()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Security overrides
        if 'KINICH_JWT_SECRET' in os.environ:
            self._config.setdefault('security', {})['jwt_secret'] = os.environ['KINICH_JWT_SECRET']
        
        if 'KINICH_ENABLE_AUTH' in os.environ:
            self._config.setdefault('security', {})['enable_authentication'] = \
                os.environ['KINICH_ENABLE_AUTH'].lower() == 'true'
        
        # Quantum overrides
        if 'KINICH_DEFAULT_BACKEND' in os.environ:
            self._config.setdefault('quantum', {})['default_backend'] = \
                os.environ['KINICH_DEFAULT_BACKEND']
        
        if 'KINICH_MAX_QUBITS' in os.environ:
            self._config.setdefault('quantum', {})['max_qubits'] = \
                int(os.environ['KINICH_MAX_QUBITS'])
        
        # Sovereignty overrides
        if 'KINICH_ENFORCE_RESIDENCY' in os.environ:
            self._config.setdefault('sovereignty', {})['enforce_data_residency'] = \
                os.environ['KINICH_ENFORCE_RESIDENCY'].lower() == 'true'
    
    # ==================== VALIDATION ====================
    
    def validate_config(self):
        """Validate configuration against schema."""
        self._validate_section(self._config, self._schema, [])
        logger.info("Configuration validated successfully")
    
    def _validate_section(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any],
        path: List[str]
    ):
        """Recursively validate configuration section."""
        for key, rules in schema.items():
            current_path = path + [key]
            field_name = '.'.join(current_path)
            
            # Check required fields
            if rules.get('required', False) and key not in config:
                raise ConfigValidationError(
                    field=field_name,
                    message=f"Required field '{field_name}' is missing"
                )
            
            if key not in config:
                continue
            
            value = config[key]
            expected_type = rules.get('type')
            
            # Type validation
            if expected_type and not isinstance(value, expected_type):
                raise ConfigValidationError(
                    field=field_name,
                    message=f"Field '{field_name}' must be {expected_type.__name__}, got {type(value).__name__}"
                )
            
            # String validation
            if isinstance(value, str):
                if 'min_length' in rules and len(value) < rules['min_length']:
                    raise ConfigValidationError(
                        field=field_name,
                        message=f"Field '{field_name}' must be at least {rules['min_length']} characters"
                    )
                
                if 'allowed' in rules and value not in rules['allowed']:
                    raise ConfigValidationError(
                        field=field_name,
                        message=f"Field '{field_name}' must be one of {rules['allowed']}"
                    )
            
            # Numeric validation
            if isinstance(value, (int, float)):
                if 'min' in rules and value < rules['min']:
                    raise ConfigValidationError(
                        field=field_name,
                        message=f"Field '{field_name}' must be >= {rules['min']}"
                    )
                
                if 'max' in rules and value > rules['max']:
                    raise ConfigValidationError(
                        field=field_name,
                        message=f"Field '{field_name}' must be <= {rules['max']}"
                    )
            
            # Nested schema validation
            if 'schema' in rules and isinstance(value, dict):
                self._validate_section(value, rules['schema'], current_path)
    
    # ==================== CONFIG ACCESS ====================
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'quantum.max_qubits')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        parts = key.split('.')
        value = self._config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        parts = key.split('.')
        config = self._config
        
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        config[parts[-1]] = value
        
        # Notify watchers
        self._notify_watchers(key, value)
    
    def get_all(self) -> Dict[str, Any]:
        """Get full configuration."""
        return self._config.copy()
    
    # ==================== SECRETS MANAGEMENT ====================
    
    def load_secrets(self, secrets_path: Optional[str] = None):
        """
        Load secrets from secure store.
        
        In production, this would integrate with:
        - HashiCorp Vault
        - AWS Secrets Manager
        - Azure Key Vault
        - Kubernetes Secrets
        
        Args:
            secrets_path: Path to secrets file (for development)
        """
        if secrets_path:
            try:
                with open(secrets_path, 'r') as f:
                    self._secrets = json.load(f)
                logger.info(f"Loaded {len(self._secrets)} secrets")
            except Exception as e:
                logger.error(f"Failed to load secrets: {e}")
        else:
            # Load from environment variables
            self._secrets = {
                'jwt_secret': os.environ.get('KINICH_JWT_SECRET', ''),
                'azure_quantum_key': os.environ.get('AZURE_QUANTUM_KEY', ''),
                'ibm_quantum_token': os.environ.get('IBM_QUANTUM_TOKEN', ''),
            }
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret value."""
        return self._secrets.get(key)
    
    # ==================== HOT RELOAD ====================
    
    def watch_config(self, callback: Callable[[str, Any], None]):
        """Register callback for configuration changes."""
        self._watchers.append(callback)
    
    def _notify_watchers(self, key: str, value: Any):
        """Notify watchers of configuration change."""
        for watcher in self._watchers:
            try:
                watcher(key, value)
            except Exception as e:
                logger.error(f"Config watcher failed: {e}")
    
    def reload(self):
        """Reload configuration from file."""
        if self.config_path:
            logger.info("Reloading configuration...")
            self.load_config(self.config_path)
            
            # Notify all watchers
            for key, value in self._config.items():
                self._notify_watchers(key, value)
    
    # ==================== EXPORT ====================
    
    def export_yaml(self, output_path: str):
        """Export configuration to YAML file."""
        try:
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
            logger.info(f"Exported configuration to {output_path}")
        except ImportError:
            logger.error("PyYAML not installed")
    
    def export_json(self, output_path: str):
        """Export configuration to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self._config, f, indent=2)
        logger.info(f"Exported configuration to {output_path}")
