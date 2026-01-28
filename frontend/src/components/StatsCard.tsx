interface Props {
    icon: string;
    label: string;
    value: string | number;
    trend?: 'up' | 'down';
    color?: 'blue' | 'yellow' | 'red' | 'green';
}

export default function StatsCard({ icon, label, value, trend, color = 'blue' }: Props) {
    return (
        <div className="stat-card">
            <div className={`stat-icon ${color}`}>
                {icon}
            </div>
            <div className="stat-content">
                <div className="stat-label">{label}</div>
                <div className="stat-value">
                    {value}
                    {trend && (
                        <span style={{ fontSize: '0.6em', marginLeft: '0.5rem' }}>
                            {trend === 'up' ? '📈' : '📉'}
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
}
