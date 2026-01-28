import { Transaction } from '../types';

interface Props {
    transaction: Transaction;
    onClick?: () => void;
}

export default function TransactionCard({ transaction, onClick }: Props) {
    const getRiskLevel = (score: number): 'low' | 'medium' | 'high' => {
        if (score < 25) return 'low';
        if (score < 60) return 'medium';
        return 'high';
    };

    const getStatusBadgeClass = (status: string) => {
        switch (status) {
            case 'approved':
                return 'badge-success';
            case 'flagged':
                return 'badge-warning';
            case 'blocked':
                return 'badge-danger';
            default:
                return 'badge-info';
        }
    };

    return (
        <div
            className={`transaction-card ${transaction.status}`}
            onClick={onClick}
        >
            <div className="transaction-header">
                <div>
                    <div className="transaction-amount">${transaction.amount.toFixed(2)}</div>
                    <div className="transaction-meta">
                        <div>📍 {transaction.merchant}</div>
                        <div>🌐 {transaction.location}</div>
                        <div>🕐 {new Date(transaction.timestamp).toLocaleTimeString()}</div>
                    </div>
                </div>
                <div>
                    <span className={`badge ${getStatusBadgeClass(transaction.status)}`}>
                        {transaction.status}
                    </span>
                </div>
            </div>

            <div className="risk-indicator">
                <div className="risk-label">
                    Risk: {transaction.risk_score.toFixed(1)}%
                </div>
                <div className="risk-bar">
                    <div
                        className={`risk-fill ${getRiskLevel(transaction.risk_score)}`}
                        style={{ width: `${Math.min(transaction.risk_score, 100)}%` }}
                    />
                </div>
            </div>
        </div>
    );
}
