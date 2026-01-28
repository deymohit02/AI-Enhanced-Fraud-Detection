// TypeScript interfaces and types for FraudGuard AI

export interface Transaction {
    id: string;
    amount: number;
    merchant: string;
    location: string;
    timestamp: string;
    risk_score: number;
    status: 'approved' | 'flagged' | 'blocked';
    predictions: {
        rules_engine: number;
        ml_model: number;
        dl_model: number;
    };
    active_model: string;
    is_actual_fraud?: boolean;
}

export interface Stats {
    total_transactions: number;
    flagged_transactions: number;
    blocked_transactions: number;
    avg_risk_score: number;
}

export interface ModelMetrics {
    name: string;
    version: string;
    total_predictions: number;
    fraud_detected: number;
    fraud_rate: number;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    false_positive_rate: number;
    false_negative_rate: number;
    is_active?: boolean;
    model_loaded?: boolean;
    tf_available?: boolean;
}

export interface Model {
    id: string;
    name: string;
    version: string;
    is_active: boolean;
    metrics: ModelMetrics;
}

export interface ChartDataset {
    label: string;
    data: number[];
    color: string;
}

export interface ChartData {
    labels: string[];
    datasets: ChartDataset[];
}

export interface HistoricalData {
    accuracy_trends: ChartData;
    false_positive_rates: ChartData;
}

export interface ModelComparison {
    models: Model[];
    historical_data: HistoricalData;
    best_model: {
        id: string;
        name: string;
        version: string;
        reason: string;
        recommendation: string;
    };
}

export interface AIExplanation {
    explanation: string;
    similar_cases: Array<{
        description: string;
        distance: number;
        metadata: any;
    }>;
    transaction_id: string;
}

export interface WebSocketMessage {
    type: 'transaction' | 'stats' | 'status' | 'pong';
    data?: any;
    message?: string;
    monitoring?: boolean;
}
