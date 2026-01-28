import { useState } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import StatsCard from '../components/StatsCard';
import TransactionCard from '../components/TransactionCard';
import { Transaction, AIExplanation } from '../types';

export default function LiveDashboard() {
    const {
        transactions,
        stats,
        isConnected,
        isMonitoring,
        startMonitoring,
        stopMonitoring
    } = useWebSocket();

    const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
    const [aiExplanation, setAiExplanation] = useState<AIExplanation | null>(null);
    const [isExplaining, setIsExplaining] = useState(false);

    const handleMonitoringToggle = () => {
        if (isMonitoring) {
            stopMonitoring();
        } else {
            startMonitoring();
        }
    };

    const handleExplain = async (tx: Transaction) => {
        setIsExplaining(true);
        setAiExplanation(null);
        try {
            const response = await fetch('http://localhost:8000/api/explain', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    transaction_id: tx.id,
                    amount: tx.amount,
                    merchant: tx.merchant,
                    location: tx.location,
                    risk_score: tx.risk_score
                }),
            });
            const data = await response.json();
            setAiExplanation(data);
        } catch (error) {
            console.error('Error fetching AI explanation:', error);
        } finally {
            setIsExplaining(false);
        }
    };

    const handleTransactionSelect = (tx: Transaction) => {
        setSelectedTransaction(tx);
        setAiExplanation(null); // Reset explanation when selecting a new transaction
    };

    return (
        <div className="container">
            <div className="page-header">
                <h1 className="page-title">Real-Time Fraud Detection Dashboard</h1>
            </div>

            {/* Stats Grid */}
            <div className="stats-grid">
                <StatsCard
                    icon="📊"
                    label="Total Transactions"
                    value={stats.total_transactions}
                    trend="up"
                    color="blue"
                />
                <StatsCard
                    icon="⚠️"
                    label="Flagged Transactions"
                    value={stats.flagged_transactions}
                    color="yellow"
                />
                <StatsCard
                    icon="🛡️"
                    label="Blocked Transactions"
                    value={stats.blocked_transactions}
                    color="red"
                />
                <StatsCard
                    icon="💰"
                    label="Avg Risk Score"
                    value={`${stats.avg_risk_score.toFixed(1)}%`}
                    color="green"
                />
            </div>

            {/* Control Button */}
            <div className="mb-3">
                <button
                    className={`btn btn-lg ${isMonitoring ? 'btn-danger' : 'btn-success'}`}
                    onClick={handleMonitoringToggle}
                    disabled={!isConnected}
                >
                    {isMonitoring ? '🛑 Stop Monitoring' : '▶️ Start Monitoring'}
                </button>

                {!isConnected && (
                    <span style={{ marginLeft: '1rem', color: 'var(--color-danger)' }}>
                        ⚠️ Disconnected - Attempting to reconnect...
                    </span>
                )}
            </div>

            {/* Info Banner */}
            {isMonitoring && (
                <div className="info-banner mb-3">
                    ℹ️ Live monitoring active. New transactions appear every 2 seconds.
                </div>
            )}

            {/* Main Content Grid */}
            <div className="grid grid-2">
                {/* Left Panel - Live Transaction Feed */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Live Transaction Feed</h3>
                        <p className="card-subtitle">
                            {transactions.length === 0
                                ? 'Click "Start Monitoring" to begin tracking transactions'
                                : `Showing ${transactions.length} recent transactions`
                            }
                        </p>
                    </div>

                    {transactions.length === 0 ? (
                        <div className="empty-state">
                            <div className="empty-state-icon">📊</div>
                            <p className="empty-state-text">
                                No transactions yet. Start monitoring to see real-time fraud detection in action.
                            </p>
                        </div>
                    ) : (
                        <div className="transaction-list">
                            {transactions.map((transaction) => (
                                <TransactionCard
                                    key={transaction.id}
                                    transaction={transaction}
                                    onClick={() => handleTransactionSelect(transaction)}
                                />
                            ))}
                        </div>
                    )}
                </div>

                {/* Right Panel - Transaction Analysis */}
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title">Transaction Analysis</h3>
                        <p className="card-subtitle">
                            Select a transaction from the feed to see detailed analysis
                        </p>
                    </div>

                    {!selectedTransaction ? (
                        <div className="empty-state">
                            <div className="empty-state-icon">🔍</div>
                            <p className="empty-state-text">
                                Select a transaction from the feed to see detailed analysis
                            </p>
                        </div>
                    ) : (
                        <div>
                            {/* Transaction Details */}
                            <div className="mb-3">
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--spacing-md)' }}>
                                    <h4 style={{ fontSize: 'var(--font-size-lg)', marginBottom: 0 }}>
                                        Transaction Details
                                    </h4>
                                    <button
                                        className="btn btn-sm btn-info"
                                        onClick={() => handleExplain(selectedTransaction)}
                                        disabled={isExplaining}
                                        style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                                    >
                                        {isExplaining ? '⏳ Analyzing...' : '✨ Explain with AI'}
                                    </button>
                                </div>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-md)' }}>
                                    <div>
                                        <label style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)' }}>
                                            Transaction ID
                                        </label>
                                        <div>{selectedTransaction.id}</div>
                                    </div>

                                    <div>
                                        <label style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)' }}>
                                            Amount
                                        </label>
                                        <div style={{ fontSize: 'var(--font-size-lg)', fontWeight: 700 }}>
                                            ${selectedTransaction.amount.toFixed(2)}
                                        </div>
                                    </div>

                                    <div>
                                        <label style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)' }}>
                                            Merchant
                                        </label>
                                        <div>{selectedTransaction.merchant}</div>
                                    </div>

                                    <div>
                                        <label style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)' }}>
                                            Location
                                        </label>
                                        <div>{selectedTransaction.location}</div>
                                    </div>
                                </div>

                                <div className="mt-3">
                                    <label style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: 'var(--font-size-sm)' }}>
                                        Status
                                    </label>
                                    <div className="mt-1">
                                        <span className={`badge ${selectedTransaction.status === 'approved' ? 'badge-success' :
                                            selectedTransaction.status === 'flagged' ? 'badge-warning' :
                                                'badge-danger'
                                            }`}>
                                            {selectedTransaction.status.toUpperCase()}
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* AI Analysis Section */}
                            {aiExplanation && (
                                <div className="mt-4 animate-slide-in" style={{
                                    padding: 'var(--spacing-md)',
                                    background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                                    borderRadius: 'var(--border-radius)',
                                    borderLeft: '4px solid var(--color-info)'
                                }}>
                                    <h4 style={{ fontSize: 'var(--font-size-md)', color: 'var(--color-info)', marginBottom: 'var(--spacing-sm)', display: 'flex', alignItems: 'center' }}>
                                        ✨ AI Investigation Report
                                    </h4>
                                    <div style={{
                                        fontSize: 'var(--font-size-sm)',
                                        lineHeight: '1.6',
                                        whiteSpace: 'pre-line',
                                        color: 'var(--text-primary)'
                                    }}>
                                        {aiExplanation.explanation}
                                    </div>

                                    {aiExplanation.similar_cases && aiExplanation.similar_cases.length > 0 && (
                                        <div className="mt-3">
                                            <p style={{ fontWeight: 600, fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 'var(--spacing-xs)' }}>
                                                Similar Historical Patterns (RAG)
                                            </p>
                                            {aiExplanation.similar_cases.map((cc, idx) => (
                                                <div key={idx} style={{
                                                    fontSize: 'var(--font-size-xs)',
                                                    padding: 'var(--spacing-xs)',
                                                    background: 'rgba(255,255,255,0.5)',
                                                    borderRadius: '4px',
                                                    marginBottom: '2px'
                                                }}>
                                                    🔍 {cc.description}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}

                            {isExplaining && (
                                <div className="mt-4 p-3 text-center" style={{ background: '#f8fafc', borderRadius: 'var(--border-radius)' }}>
                                    <div className="spinner mb-2"></div>
                                    <p className="text-secondary small">Gemini is analyzing transaction patterns...</p>
                                </div>
                            )}

                            {/* Model Predictions */}
                            <div className="mt-4">
                                <h4 style={{ fontSize: 'var(--font-size-md)', marginBottom: 'var(--spacing-sm)' }}>
                                    Model Confidence Scores
                                </h4>
                                <div style={{ display: 'grid', gap: 'var(--spacing-xs)' }}>
                                    <div style={{
                                        padding: 'var(--spacing-sm) var(--spacing-md)',
                                        background: 'var(--color-gray-50)',
                                        borderRadius: 'var(--border-radius)',
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        fontSize: 'var(--font-size-sm)'
                                    }}>
                                        <span>Rules Engine</span>
                                        <span style={{ fontWeight: 700 }}>
                                            {selectedTransaction.predictions.rules_engine.toFixed(1)}%
                                        </span>
                                    </div>

                                    <div style={{
                                        padding: 'var(--spacing-sm) var(--spacing-md)',
                                        background: selectedTransaction.active_model === 'ml_model' ? '#dbeafe' : 'var(--color-gray-50)',
                                        borderRadius: 'var(--border-radius)',
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        fontSize: 'var(--font-size-sm)',
                                        border: selectedTransaction.active_model === 'ml_model' ? '1px solid var(--color-info)' : 'none'
                                    }}>
                                        <span>
                                            ML Model (XGBoost)
                                            {selectedTransaction.active_model === 'ml_model' && (
                                                <span className="badge badge-info" style={{ marginLeft: '0.5rem', fontSize: '10px' }}>
                                                    ACTIVE
                                                </span>
                                            )}
                                        </span>
                                        <span style={{ fontWeight: 700 }}>
                                            {selectedTransaction.predictions.ml_model.toFixed(1)}%
                                        </span>
                                    </div>

                                    <div style={{
                                        padding: 'var(--spacing-sm) var(--spacing-md)',
                                        background: selectedTransaction.active_model === 'dl_model' ? '#ede9fe' : 'var(--color-gray-50)',
                                        borderRadius: 'var(--border-radius)',
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        fontSize: 'var(--font-size-sm)',
                                        border: selectedTransaction.active_model === 'dl_model' ? '1px solid var(--color-secondary)' : 'none'
                                    }}>
                                        <span>
                                            DL Model (LSTM)
                                            {selectedTransaction.active_model === 'dl_model' && (
                                                <span className="badge badge-secondary" style={{ marginLeft: '0.5rem', fontSize: '10px' }}>
                                                    ACTIVE
                                                </span>
                                            )}
                                        </span>
                                        <span style={{ fontWeight: 700 }}>
                                            {selectedTransaction.predictions.dl_model.toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Risk Assessment */}
                            <div className="mt-4">
                                <h4 style={{ fontSize: 'var(--font-size-md)', marginBottom: 'var(--spacing-sm)' }}>
                                    Composite Risk Assessment
                                </h4>
                                <div style={{
                                    padding: 'var(--spacing-md)',
                                    background: selectedTransaction.risk_score > 60 ? '#fee2e2' :
                                        selectedTransaction.risk_score > 25 ? '#fef3c7' : '#d1fae5',
                                    borderRadius: 'var(--border-radius)'
                                }}>
                                    <div style={{ fontSize: 'var(--font-size-lg)', fontWeight: 700, marginBottom: 'var(--spacing-sm)' }}>
                                        {selectedTransaction.risk_score.toFixed(1)}% Risk
                                    </div>
                                    <div className="risk-bar" style={{ height: '8px' }}>
                                        <div
                                            className={`risk-fill ${selectedTransaction.risk_score < 25 ? 'low' :
                                                selectedTransaction.risk_score < 60 ? 'medium' : 'high'
                                                }`}
                                            style={{ width: `${Math.min(selectedTransaction.risk_score, 100)}%` }}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
