import { useState, useEffect } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { ModelComparison as ModelComparisonType } from '../types';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend
);

const API_BASE = 'http://localhost:8000';

export default function ModelComparison() {
    const [comparisonData, setComparisonData] = useState<ModelComparisonType | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchComparisonData();
    }, []);

    const fetchComparisonData = async () => {
        try {
            const response = await fetch(`${API_BASE}/api/models/comparison`);
            const data = await response.json();
            setComparisonData(data);
            setLoading(false);
        } catch (err) {
            setError('Failed to load model comparison data');
            setLoading(false);
        }
    };

    const activateModel = async (modelId: string) => {
        try {
            await fetch(`${API_BASE}/api/models/activate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId }),
            });

            // Refresh data
            fetchComparisonData();
        } catch (err) {
            console.error('Failed to activate model:', err);
        }
    };

    if (loading) {
        return (
            <div className="container">
                <div className="empty-state" style={{ paddingTop: '4rem' }}>
                    <div className="empty-state-icon">⏳</div>
                    <p className="empty-state-text">Loading model comparison data...</p>
                </div>
            </div>
        );
    }

    if (error || !comparisonData) {
        return (
            <div className="container">
                <div className="empty-state" style={{ paddingTop: '4rem' }}>
                    <div className="empty-state-icon">❌</div>
                    <p className="empty-state-text">{error || 'Failed to load data'}</p>
                </div>
            </div>
        );
    }

    // Prepare chart data for accuracy trends
    const accuracyChartData = {
        labels: comparisonData.historical_data.accuracy_trends.labels,
        datasets: comparisonData.historical_data.accuracy_trends.datasets.map((dataset) => ({
            label: dataset.label,
            data: dataset.data,
            borderColor: dataset.color,
            backgroundColor: dataset.color + '20',
            tension: 0.3,
        })),
    };

    // Prepare chart data for false positive rates
    const fprChartData = {
        labels: comparisonData.historical_data.false_positive_rates.labels,
        datasets: comparisonData.historical_data.false_positive_rates.datasets.map((dataset) => ({
            label: dataset.label,
            data: dataset.data,
            backgroundColor: dataset.color,
        })),
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: {
                position: 'bottom' as const,
            },
        },
        scales: {
            y: {
                beginAtZero: false,
            },
        },
    };

    return (
        <div className="container">
            <div className="page-header">
                <h1 className="page-title">Model Performance Comparison</h1>
                <p className="page-subtitle">
                    Compare different fraud detection models and their performance metrics
                </p>
            </div>

            {/* Model Cards */}
            <div className="grid-3 mb-4">
                {comparisonData.models.map((model) => (
                    <div key={model.id} className="card" style={{ position: 'relative' }}>
                        <div style={{ marginBottom: 'var(--spacing-md)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                <div>
                                    <h3 className="card-title">{model.name}</h3>
                                    <p className="card-subtitle">Version {model.version}</p>
                                </div>
                                {model.is_active && (
                                    <span className="badge badge-info">ACTIVE</span>
                                )}
                            </div>
                        </div>

                        {/* Main Metrics */}
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 1fr',
                            gap: 'var(--spacing-md)',
                            marginBottom: 'var(--spacing-lg)'
                        }}>
                            <div style={{ textAlign: 'center' }}>
                                <div style={{
                                    fontSize: 'var(--font-size-3xl)',
                                    fontWeight: 700,
                                    color: 'var(--color-success)'
                                }}>
                                    {(model.metrics.accuracy * 100).toFixed(1)}%
                                </div>
                                <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)' }}>
                                    Accuracy
                                </div>
                            </div>

                            <div style={{ textAlign: 'center' }}>
                                <div style={{
                                    fontSize: 'var(--font-size-3xl)',
                                    fontWeight: 700,
                                    color: 'var(--color-danger)'
                                }}>
                                    {(model.metrics.false_positive_rate * 100).toFixed(1)}%
                                </div>
                                <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)' }}>
                                    False Positive Rate
                                </div>
                            </div>
                        </div>

                        {/* Secondary Metrics */}
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 1fr',
                            gap: 'var(--spacing-md)',
                            marginBottom: 'var(--spacing-lg)'
                        }}>
                            <div style={{ textAlign: 'center' }}>
                                <div style={{
                                    fontSize: 'var(--font-size-xl)',
                                    fontWeight: 700,
                                    color: 'var(--color-info)'
                                }}>
                                    {(model.metrics.precision * 100).toFixed(1)}%
                                </div>
                                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
                                    Precision
                                </div>
                            </div>

                            <div style={{ textAlign: 'center' }}>
                                <div style={{
                                    fontSize: 'var(--font-size-xl)',
                                    fontWeight: 700,
                                    color: '#a855f7'
                                }}>
                                    {(model.metrics.recall * 100).toFixed(1)}%
                                </div>
                                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
                                    Recall
                                </div>
                            </div>
                        </div>

                        {/* Additional Info */}
                        <div style={{
                            padding: 'var(--spacing-md)',
                            background: 'var(--color-gray-50)',
                            borderRadius: 'var(--border-radius)',
                            fontSize: 'var(--font-size-sm)',
                            marginBottom: 'var(--spacing-md)'
                        }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                <span>Risk Threshold:</span>
                                <span style={{ fontWeight: 600 }}>50%</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span>F1 Score:</span>
                                <span style={{ fontWeight: 600 }}>{(model.metrics.f1_score * 100).toFixed(1)}%</span>
                            </div>
                        </div>

                        {/* Activate Button */}
                        {!model.is_active && (
                            <button
                                className="btn btn-outline"
                                onClick={() => activateModel(model.id)}
                                style={{ width: '100%' }}
                            >
                                Activate
                            </button>
                        )}
                    </div>
                ))}
            </div>

            {/* Charts Section */}
            <div className="grid-2 mb-4">
                {/* Accuracy Trends */}
                <div className="card">
                    <h3 className="card-title">📈 Accuracy Trends</h3>
                    <div style={{ padding: 'var(--spacing-md) 0' }}>
                        <Line data={accuracyChartData} options={chartOptions} />
                    </div>
                </div>

                {/* False Positive Rates */}
                <div className="card">
                    <h3 className="card-title">🚨 False Positive Rates</h3>
                    <div style={{ padding: 'var(--spacing-md) 0' }}>
                        <Bar data={fprChartData} options={chartOptions} />
                    </div>
                </div>
            </div>

            {/* Model Comparison Table */}
            <div className="card mb-4">
                <h3 className="card-title">📊 Model Comparison Summary</h3>

                <div style={{ overflowX: 'auto' }}>
                    <table style={{
                        width: '100%',
                        borderCollapse: 'collapse',
                        marginTop: 'var(--spacing-md)'
                    }}>
                        <thead>
                            <tr style={{
                                borderBottom: '2px solid var(--border-color)',
                                textAlign: 'left'
                            }}>
                                <th style={{ padding: 'var(--spacing-md)', fontWeight: 600 }}>Model</th>
                                <th style={{ padding: 'var(--spacing-md)', fontWeight: 600 }}>Version</th>
                                <th style={{ padding: 'var(--spacing-md)', fontWeight: 600 }}>Accuracy</th>
                                <th style={{ padding: 'var(--spacing-md)', fontWeight: 600 }}>False Positive Rate</th>
                                <th style={{ padding: 'var(--spacing-md)', fontWeight: 600 }}>False Negative Rate</th>
                                <th style={{ padding: 'var(--spacing-md)', fontWeight: 600 }}>Precision</th>
                                <th style={{ padding: 'var(--spacing-md)', fontWeight: 600 }}>Recall</th>
                                <th style={{ padding: 'var(--spacing-md)', fontWeight: 600 }}>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {comparisonData.models.map((model) => (
                                <tr key={model.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                    <td style={{ padding: 'var(--spacing-md)' }}>{model.name}</td>
                                    <td style={{ padding: 'var(--spacing-md)' }}>{model.version}</td>
                                    <td style={{ padding: 'var(--spacing-md)' }}>{(model.metrics.accuracy * 100).toFixed(1)}%</td>
                                    <td style={{ padding: 'var(--spacing-md)' }}>{(model.metrics.false_positive_rate * 100).toFixed(1)}%</td>
                                    <td style={{ padding: 'var(--spacing-md)' }}>{(model.metrics.false_negative_rate * 100).toFixed(1)}%</td>
                                    <td style={{ padding: 'var(--spacing-md)' }}>{(model.metrics.precision * 100).toFixed(1)}%</td>
                                    <td style={{ padding: 'var(--spacing-md)' }}>{(model.metrics.recall * 100).toFixed(1)}%</td>
                                    <td style={{ padding: 'var(--spacing-md)' }}>
                                        {model.is_active ? (
                                            <span className="badge badge-success">Active</span>
                                        ) : (
                                            <span className="badge" style={{ background: 'var(--color-gray-200)', color: 'var(--text-secondary)' }}>
                                                Inactive
                                            </span>
                                        )}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Performance Alert */}
            {comparisonData.best_model && (
                <div className="warning-banner">
                    <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>
                        ⚠️ Model Performance Analysis
                    </div>
                    <div>
                        {comparisonData.best_model.name} v{comparisonData.best_model.version} shows the best overall performance with {comparisonData.best_model.reason}.
                    </div>
                    <div style={{ marginTop: '0.5rem', fontSize: 'var(--font-size-sm)' }}>
                        {comparisonData.best_model.recommendation}
                    </div>
                </div>
            )}
        </div>
    );
}
