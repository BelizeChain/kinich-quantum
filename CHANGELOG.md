# Changelog

All notable changes to Kinich Quantum will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial standalone repository setup
- Modern `pyproject.toml` packaging configuration
- Comprehensive README with architecture and examples
- MIT License and CONTRIBUTING guidelines

### Changed
- Removed development terminology ("Phase 1", "Phase 1.5") from codebase
- Improved docstrings to industry standards
- Migrated from `setup.py` to `pyproject.toml`

## [0.1.0] - 2026-01-26

### Added
- Production-hardened quantum node with security, sovereignty, and monitoring
- Multi-backend support (Azure Quantum, IBM Quantum, Google Cirq, SpinQ)
- Error mitigation (readout correction, ZNE, symmetry verification)
- Security manager with JWT/API key authentication and RBAC
- Sovereignty manager with data residency enforcement
- Monitoring manager with distributed tracing and metrics
- Configuration manager with YAML support
- Job scheduler with priority queuing
- Circuit optimizer with backend-specific transpilation
- Result aggregator for multi-backend execution
- Blockchain integration for Proof of Quantum Work

### Security
- Implemented role-based access control (RBAC)
- Added audit logging for all operations
- Encrypted sensitive data at rest and in transit
- API key and JWT token authentication

---

## Release Notes

### v0.1.0 - Initial Production Release

This is the first production-ready release of Kinich Quantum, extracted from the BelizeChain monorepo as a standalone package.

**Highlights:**
- ✅ Production-grade security with authentication and authorization
- ✅ Multi-backend quantum execution (Azure, IBM, Google, SpinQ)
- ✅ Comprehensive error mitigation for noisy quantum hardware
- ✅ Data sovereignty controls for regulatory compliance
- ✅ Enterprise monitoring with Prometheus and distributed tracing

**Breaking Changes:**
- None (initial release)

**Migration Guide:**
- Install via `pip install kinich-quantum`
- Update imports from `belizechain.kinich` to `kinich`
- Configuration files moved from `config/kinich.yaml` to `kinich/config/`

**Known Issues:**
- Qiskit adapters temporarily disabled pending API migration
- ZNE error mitigation is computationally expensive (use sparingly)
- IBM Quantum backend requires manual token configuration

**Contributors:**
- BelizeChain Core Team

[Unreleased]: https://github.com/BelizeChain/kinich-quantum/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BelizeChain/kinich-quantum/releases/tag/v0.1.0
