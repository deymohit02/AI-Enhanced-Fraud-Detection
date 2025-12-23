-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Transactions Table
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    amount DECIMAL(10, 2) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    features JSONB,  -- Store all input features as JSON
    is_fraud BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Predictions Table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id UUID REFERENCES transactions(id),
    model_name VARCHAR(50) NOT NULL,
    fraud_probability DECIMAL(5, 4),
    prediction BOOLEAN,
    explanation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Fraud Patterns Table (for RAG)
CREATE TABLE IF NOT EXISTS fraud_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    description TEXT,
    features JSONB,
    embedding_vector TEXT, -- Store as string or vector type if using pgvector
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions(transaction_id);
