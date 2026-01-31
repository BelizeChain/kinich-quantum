# Contributing to Kinich

Thank you for your interest in contributing to Kinich! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate in your communications
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Basic understanding of quantum computing concepts
- Familiarity with async Python programming

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/kinich-quantum.git
cd kinich-quantum

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

## Development Workflow

1. **Create an Issue**: Before starting work, create or find an issue describing what you want to do
2. **Create a Branch**: Use descriptive branch names (e.g., `feature/add-rigetti-backend`, `fix/error-mitigation-bug`)
3. **Make Changes**: Follow our code standards (see below)
4. **Write Tests**: All new features require tests
5. **Update Documentation**: Update README, docstrings, and examples as needed
6. **Commit Changes**: Use clear, descriptive commit messages
7. **Submit Pull Request**: Reference the issue number in your PR description

### Branch Naming Convention

- `feature/description`: New features
- `fix/description`: Bug fixes
- `docs/description`: Documentation improvements
- `refactor/description`: Code refactoring
- `test/description`: Test improvements

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Example:**
```
feat(adapters): Add support for Rigetti quantum backend

Implements QCS adapter for Rigetti quantum processors with
error mitigation and calibration support.

Closes #42
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no functional changes)
- `refactor`: Code restructuring
- `test`: Test additions or modifications
- `chore`: Maintenance tasks

## Code Standards

### Python Style Guide

We follow PEP 8 with some modifications enforced by Ruff:

- **Line length**: 100 characters (not 80)
- **Quotes**: Double quotes for strings
- **Imports**: Organized by isort (standard library, third-party, first-party)
- **Type hints**: Use type hints for all function signatures
- **Docstrings**: Google-style docstrings for all public APIs

### Example Function

```python
from typing import Dict, List, Optional
import asyncio


async def process_quantum_job(
    job_id: str,
    circuit: QuantumCircuit,
    backend: str,
    shots: int = 1024,
    options: Optional[Dict[str, Any]] = None
) -> QuantumJobResult:
    """Process a quantum job on the specified backend.
    
    Args:
        job_id: Unique identifier for the job
        circuit: Quantum circuit to execute
        backend: Name of the quantum backend (e.g., 'ionq.simulator')
        shots: Number of measurement shots (default: 1024)
        options: Optional backend-specific configuration
        
    Returns:
        QuantumJobResult containing execution results and metadata
        
    Raises:
        BackendNotAvailableError: If the specified backend is unavailable
        CircuitTooLargeError: If circuit exceeds backend qubit limits
        
    Examples:
        >>> result = await process_quantum_job(
        ...     job_id="job_123",
        ...     circuit=bell_circuit,
        ...     backend="ionq.simulator",
        ...     shots=2048
        ... )
        >>> print(result.counts)
        {'00': 1024, '11': 1024}
    """
    if options is None:
        options = {}
    
    # Implementation...
```

### Code Quality Checks

Before submitting, ensure your code passes all checks:

```bash
# Format code
ruff format kinich tests

# Lint code
ruff check kinich tests --fix

# Type check
mypy kinich

# Run all pre-commit hooks
pre-commit run --all-files
```

## Testing

### Writing Tests

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test component interactions
- **Quantum tests**: Mark tests requiring quantum backends with `@pytest.mark.quantum`

```python
import pytest
from kinich import QuantumNode
from kinich.core import QuantumJob


class TestQuantumNode:
    """Test suite for QuantumNode class."""
    
    def test_node_initialization(self):
        """Test that node initializes with correct default config."""
        node = QuantumNode(node_id="test_node")
        assert node.node_id == "test_node"
        assert node.status == "initialized"
    
    @pytest.mark.asyncio
    async def test_job_submission(self):
        """Test job submission workflow."""
        node = QuantumNode()
        job = QuantumJob(circuit=bell_circuit, backend="simulator")
        result = await node.submit_job(job)
        assert result.success is True
    
    @pytest.mark.quantum
    @pytest.mark.slow
    async def test_azure_backend_execution(self):
        """Test execution on Azure Quantum backend."""
        # Requires Azure credentials
        ...
```

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests (fast)
pytest -m "unit"

# Run without quantum backend tests
pytest -m "not quantum"

# Run with coverage
pytest --cov=kinich --cov-report=html

# Run specific test file
pytest tests/test_quantum_node.py -v
```

## Pull Request Process

1. **Update Documentation**: Ensure README and docstrings are current
2. **Add Tests**: All new features must have tests (aim for >80% coverage)
3. **Pass CI Checks**: All GitHub Actions workflows must pass
4. **Request Review**: Tag relevant maintainers for review
5. **Address Feedback**: Respond to review comments promptly
6. **Squash Commits**: Before merging, squash commits into logical units

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated (README, docstrings, examples)
- [ ] Commit messages follow conventional format
- [ ] No merge conflicts with main branch
- [ ] PR description clearly explains changes

## Architecture Guidelines

### Adding New Quantum Backends

1. Create adapter in `kinich/adapters/your_backend.py`
2. Implement `QuantumBackendAdapter` interface
3. Add backend configuration to `config/backends.yaml`
4. Write tests in `tests/adapters/test_your_backend.py`
5. Update documentation with backend support

### Adding New Error Mitigation Techniques

1. Implement technique in `kinich/error_mitigation/`
2. Add configuration options to `ErrorMitigationConfig`
3. Write unit tests with simulated noise
4. Benchmark performance impact
5. Document in `docs/error-mitigation.md`

## Community

- **GitHub Discussions**: Ask questions, share ideas
- **Discord**: Real-time chat with maintainers and contributors
- **Monthly Calls**: Open community calls (first Tuesday of each month)
- **Issue Tracker**: Report bugs, request features

### Getting Help

- Check existing issues and discussions
- Read the documentation at https://docs.belizechain.org/kinich
- Ask in Discord: https://discord.gg/belizechain
- Email: quantum@belizechain.org

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Featured on the project website (with permission)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Kinich!** Your efforts help advance quantum computing for everyone. 🚀
