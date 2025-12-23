-- AI-Enhanced Fraud Detection Database Schema
-- PostgreSQL (Supabase)

-- ==================================
-- 1. USERS TABLE
-- ==================================
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    account_age_days INT DEFAULT 0,
    total_transactions INT DEFAULT 0,
    avg_transaction_amount DECIMAL(10, 2) DEFAULT 0.00,
    risk_score DECIMAL(5, 4) DEFAULT 0.0000,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==================================
-- 2. MERCHANTS TABLE
-- ==================================
CREATE TABLE IF NOT EXISTS merchants (
    merchant_id VARCHAR(50) PRIMARY KEY,
    merchant_name VARCHAR(255),
    category VARCHAR(100),
    risk_level VARCHAR(20) DEFAULT 'LOW', -- LOW, MEDIUM, HIGH
    fraud_rate DECIMAL(5, 4)DEFAULT 0.0000,
    total_transactions INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==================================
-- 3. TRANSACTIONS TABLE (Enhanced)
-- ==================================
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Basic Transaction Info
    amount DECIMAL(12, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- User & Card Info
    user_id VARCHAR(50),
    card_id VARCHAR(50),
    
    -- Merchant Info
    merchant_id VARCHAR(50),
    merchant_category VARCHAR(100),
    
    -- Location Info
    location_country VARCHAR(100),
    location_city VARCHAR(100),
    ip_address VARCHAR(45),
    
    -- Device Info
    device_id VARCHAR(100),
    device_fingerprint TEXT,
    device_type VARCHAR(50), -- mobile, desktop, tablet
    
    -- Channel
    channel VARCHAR(20), -- online, pos, atm
    
    -- Features (JSON for flexibility)
    features_json JSONB, -- Stores V1-V28 + engineered features
    
    -- Ground Truth (for training)
    is_fraud BOOLEAN DEFAULT NULL, -- NULL = unknown, TRUE/FALSE = confirmed
    fraud_confirmed_at TIMESTAMP,
    fraud_confirmed_by VARCHAR(100),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Keys
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
    FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id) ON DELETE SET NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant_id ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud ON transactions(is_fraud) WHERE is_fraud IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount);

-- ==================================
-- 4. PREDICTIONS TABLE
-- ==================================
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID NOT NULL,
    
    -- Model Info
    model_name VARCHAR(50) NOT NULL, -- xgboost, lstm, ensemble, etc.
    model_version VARCHAR(20) DEFAULT 'v1.0',
    
    -- Prediction Results
    fraud_probability DECIMAL(5, 4) NOT NULL, -- 0.0000 to 1.0000
    prediction BOOLEAN NOT NULL, -- TRUE = fraud, FALSE = legitimate
    confidence_score DECIMAL(5, 4), -- Model confidence
    
    -- Feature Importance (Top 10)
    feature_importance_json JSONB,
    
    -- RAG Similarityscore (if applicable)
    rag_similarity_score DECIMAL(5, 4),
    similar_fraud_count INT DEFAULT 0,
    
    -- Processing Time
    processing_time_ms INT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key
    FOREIGN KEY (transaction_id) REFERENCES transactions(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_fraud_prob ON predictions(fraud_probability DESC);

-- ==================================
-- 5. ALERTS TABLE
-- ==================================
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID NOT NULL,
    prediction_id UUID,
    
    -- Alert Details
    severity VARCHAR(20) NOT NULL, -- LOW, MEDIUM, HIGH, CRITICAL
    alert_type VARCHAR(50) DEFAULT 'FRAUD_PREDICTION', -- FRAUD_PREDICTION, VELOCITY_ANOMALY, etc.
    alert_message TEXT,
    
    -- Status
    status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, REVIEWED, ESCALATED, RESOLVED, FALSE_POSITIVE
    assigned_to VARCHAR(100),
    
    -- Resolution
    resolution_notes TEXT,
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(100),
    
    -- AI Explanation (from RAG)
    ai_explanation TEXT,
    similar_cases_json JSONB,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Keys
    FOREIGN KEY (transaction_id) REFERENCES transactions(id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE SET NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_assigned_to ON alerts(assigned_to) WHERE assigned_to IS NOT NULL;

-- ==================================
-- 6. FRAUD_PATTERNS TABLE (for RAG)
-- ==================================
CREATE TABLE IF NOT EXISTS fraud_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Pattern Description
    pattern_name VARCHAR(255),
    description TEXT NOT NULL,
    
    -- Pattern Characteristics
    pattern_type VARCHAR(50), -- velocity_attack, location_anomaly, amount_spike, etc.
    characteristics_json JSONB,
    
    -- Embedding for Similarity Search
    -- Note: Vector embeddings stored in FAISS separately for performance
    -- This table stores metadata
    embedding_id VARCHAR(100),
    
    -- Statistics
    detection_count INT DEFAULT 0,
    last_detected_at TIMESTAMP,
    
    -- Examples (transaction IDs)
    example_transactions_json JSONB,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index
CREATE INDEX IF NOT EXISTS idx_fraud_patterns_type ON fraud_patterns(pattern_type);

-- ==================================
-- 7. MODEL_METADATA TABLE
-- ==================================
CREATE TABLE IF NOT EXISTS model_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    
    -- Model Info
    model_type VARCHAR(50), -- xgboost, lstm, ann, ensemble
    framework VARCHAR(50), -- sklearn, tensorflow, pytorch
    
    -- Performance Metrics
    accuracy DECIMAL(5, 4),
    precision DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    auc_roc DECIMAL(5, 4),
    
    -- Training Info
    training_dataset_size INT,
    training_date TIMESTAMP,
    hyperparameters_json JSONB,
    
    -- Status
    is_active BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, model_version)
);

-- ==================================
-- 8. PERFORMANCE_LOGS TABLE
-- ==================================
CREATE TABLE IF NOT EXISTS performance_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Metrics
    endpoint VARCHAR(100),
    response_time_ms INT,
    status_code INT,
    
    -- Model Performance
    model_used VARCHAR(50),
    prediction_confidence DECIMAL(5, 4),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Partition by date for efficiency (optional)
-- CREATE INDEX IF NOT EXISTS idx_performance_logs_date ON performance_logs(created_at DESC);

-- ==================================
-- VIEWS FOR ANALYTICS
-- ==================================

-- Real-time Fraud Statistics
CREATE OR REPLACE VIEW fraud_statistics AS
SELECT
    DATE(created_at) as date,
    COUNT(*) as total_transactions,
    SUM(CASE WHEN is_fraud = TRUE THEN 1 ELSE 0 END) as fraud_count,
    ROUND(SUM(CASE WHEN is_fraud = TRUE THEN 1 ELSE 0 END)::DECIMAL / NULLIF(COUNT(*), 0) * 100, 2) as fraud_rate,
    SUM(amount) as total_amount,
    SUM(CASE WHEN is_fraud = TRUE THEN amount ELSE 0 END) as fraud_amount
FROM transactions
WHERE is_fraud IS NOT NULL
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Model Performance Comparison
CREATE OR REPLACE VIEW model_performance_comparison AS
SELECT
    model_name,
    model_version,
    COUNT(*) as prediction_count,
    AVG(fraud_probability) as avg_fraud_prob,
    AVG(processing_time_ms) as avg_processing_time,
    MAX(created_at) as last_used
FROM predictions
GROUP BY model_name, model_version
ORDER BY prediction_count DESC;

-- Pending Alerts Summary
CREATE OR REPLACE VIEW pending_alerts_summary AS
SELECT
    severity,
    COUNT(*) as alert_count,
    MIN(created_at) as oldest_alert,
    MAX(created_at) as newest_alert
FROM alerts
WHERE status = 'PENDING'
GROUP BY severity
ORDER BY 
    CASE severity
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH' THEN 2
        WHEN 'MEDIUM' THEN 3
        WHEN 'LOW' THEN 4
    END;

-- ==================================
-- SAMPLE DATA (for testing)
-- ==================================

-- Insert test users
INSERT INTO users (user_id, account_age_days, total_transactions, avg_transaction_amount, risk_score, is_verified)
VALUES 
    ('USER_001', 365, 150, 250.00, 0.1500, TRUE),
    ('USER_002', 90, 45, 500.00, 0.3000, TRUE),
    ('USER_003', 10, 5, 1500.00, 0.7500, FALSE)
ON CONFLICT (user_id) DO NOTHING;

-- Insert test merchants
INSERT INTO merchants (merchant_id, merchant_name, category, risk_level, fraud_rate, total_transactions)
VALUES
    ('MERCH_001', 'Amazon', 'retail', 'LOW', 0.0050, 10000),
    ('MERCH_002', 'LocalShop_XYZ', 'retail', 'MEDIUM', 0.0250, 500),
    ('MERCH_003', 'SuspiciousStore', 'digital', 'HIGH', 0.1500, 100)
ON CONFLICT (merchant_id) DO NOTHING;

-- ==================================
-- FUNCTIONS FOR ANALYTICS
-- ==================================

-- Function to calculate fraud rate for a date range
CREATE OR REPLACE FUNCTION get_fraud_rate(start_date DATE, end_date DATE)
RETURNS DECIMAL(5, 4) AS $$
DECLARE
    fraud_count INT;
    total_count INT;
BEGIN
    SELECT 
        COUNT(*) FILTER (WHERE is_fraud = TRUE),
        COUNT(*)
    INTO fraud_count, total_count
    FROM transactions
    WHERE DATE(created_at) BETWEEN start_date AND end_date
      AND is_fraud IS NOT NULL;
    
    IF total_count = 0 THEN
        RETURN 0.0000;
    END IF;
    
    RETURN (fraud_count::DECIMAL / total_count);
END;
$$ LANGUAGE plpgsql;

-- ==================================
-- TRIGGERS FOR AUTO-UPDATES
-- ==================================

-- Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_merchants_updated_at BEFORE UPDATE ON merchants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_alerts_updated_at BEFORE UPDATE ON alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==================================
-- NOTES
-- ==================================
-- 1. Run this script in your Supabase SQL editor
-- 2. Ensure UUID extension is enabled: CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- 3. For production, consider partitioning large tables by date
-- 4. Monitor index usage and add/remove as needed
-- 5. Set up proper RLS (Row Level Security) policies for Supabase
