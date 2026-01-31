-- Kinich Quantum - PostgreSQL Database Schema
-- Standalone schema for quantum job tracking and blockchain integration

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Quantum job tracking table
CREATE TABLE IF NOT EXISTS quantum_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(66) NOT NULL UNIQUE,
    submitter VARCHAR(66) NOT NULL,
    backend VARCHAR(50) NOT NULL,
    circuit_hash VARCHAR(66) NOT NULL,
    num_qubits SMALLINT NOT NULL,
    circuit_depth INT NOT NULL,
    num_shots INT NOT NULL,
    status VARCHAR(20) NOT NULL,
    result_hash VARCHAR(66),
    accuracy SMALLINT,
    submitted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX idx_quantum_submitter ON quantum_jobs(submitter);
CREATE INDEX idx_quantum_status ON quantum_jobs(status);
CREATE INDEX idx_quantum_backend ON quantum_jobs(backend);
CREATE INDEX idx_quantum_submitted_at ON quantum_jobs(submitted_at);
CREATE INDEX idx_quantum_metadata ON quantum_jobs USING gin(metadata);

-- Trigger to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_quantum_jobs_updated_at
BEFORE UPDATE ON quantum_jobs
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE quantum_jobs IS 'Tracks quantum computing jobs submitted to Kinich';
COMMENT ON COLUMN quantum_jobs.job_id IS 'Unique blockchain-verifiable job identifier';
COMMENT ON COLUMN quantum_jobs.submitter IS 'BelizeID address of job submitter';
COMMENT ON COLUMN quantum_jobs.backend IS 'Quantum backend used (azure, ibm, qasm_simulator, etc.)';
COMMENT ON COLUMN quantum_jobs.circuit_hash IS 'SHA-256 hash of quantum circuit';
COMMENT ON COLUMN quantum_jobs.status IS 'Job status: pending, running, completed, failed, cancelled';
COMMENT ON COLUMN quantum_jobs.result_hash IS 'Hash of job results for blockchain verification';
COMMENT ON COLUMN quantum_jobs.metadata IS 'Additional job metadata (error messages, resource usage, etc.)';
