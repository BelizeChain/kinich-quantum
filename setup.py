"""
Kinich: Quantum Computing Orchestration for BelizeChain

Hybrid quantum-classical workload orchestration with multi-backend support
(Azure Quantum, IBM Quantum), error mitigation, and blockchain integration.
"""

from setuptools import setup, find_packages

setup(
    name="kinich",
    version="0.1.0",
    description="Quantum computing orchestration with multi-backend support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="BelizeChain Core Team",
    author_email="dev@belizechain.org",
    url="https://github.com/BelizeChain/kinich-quantum",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.11",
    install_requires=[
        # Quantum frameworks
        "qiskit>=1.2.0",
        "qiskit-aer>=0.15.0",
        "qiskit-ibm-runtime>=0.28.0",
        "azure-quantum>=1.0.0",
        "cirq>=1.4.0",
        "pennylane>=0.38.0",
        
        # Azure integrations
        "azure-identity>=1.19.0",
        "azure-keyvault-secrets>=4.9.0",
        
        # Blockchain integration
        "substrate-interface>=1.7.9",
        
        # Data processing
        "numpy>=2.1.0",
        "scipy>=1.14.0",
        
        # Configuration & logging
        "pyyaml>=6.0.2",
        "python-dotenv>=1.0.1",
        "structlog>=24.4.0",
        "rich>=13.9.0",
        
        # Validation
        "pydantic>=2.9.0",
        "pydantic-settings>=2.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-asyncio>=0.24.0",
            "pytest-cov>=5.0.0",
            "pytest-mock>=3.14.0",
            "ruff>=0.7.0",
            "mypy>=1.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kinich-node=kinich.core.quantum_node:main",
            "kinich-submit=kinich.client.job_submitter:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: System :: Distributed Computing",
    ],
)
