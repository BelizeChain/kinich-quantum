# Kinich Quantum Computing Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy package files first for better caching
COPY pyproject.toml setup.py* README.md ./

# Install Python dependencies from pyproject.toml
RUN pip install --no-cache-dir -e .[server,dev,monitoring]

# Copy kinich code
COPY kinich/ ./kinich/

# Create results directory
RUN mkdir -p /app/results /app/logs /app/data

# Expose quantum API port
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV QUANTUM_BACKEND=qasm_simulator
ENV KINICH_API_HOST=0.0.0.0
ENV KINICH_API_PORT=8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Run the quantum service
CMD ["python", "-m", "kinich.api_server"]
