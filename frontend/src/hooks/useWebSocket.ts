import { useState, useEffect, useCallback, useRef } from 'react';
import { Transaction, Stats, WebSocketMessage } from '../types';

const WS_URL = 'ws://localhost:8000/ws/transactions';

export function useWebSocket() {
    const [transactions, setTransactions] = useState<Transaction[]>([]);
    const [stats, setStats] = useState<Stats>({
        total_transactions: 0,
        flagged_transactions: 0,
        blocked_transactions: 0,
        avg_risk_score: 0
    });
    const [isConnected, setIsConnected] = useState(false);
    const [isMonitoring, setIsMonitoring] = useState(false);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    const connect = useCallback(() => {
        try {
            const ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                console.log('✅ WebSocket connected');
                setIsConnected(true);

                // Send ping every 30 seconds to keep connection alive
                const pingInterval = setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ action: 'ping' }));
                    }
                }, 30000);

                ws.onclose = () => {
                    clearInterval(pingInterval);
                };
            };

            ws.onmessage = (event) => {
                const message: WebSocketMessage = JSON.parse(event.data);

                if (message.type === 'transaction') {
                    // Add new transaction to the list
                    setTransactions(prev => [message.data, ...prev].slice(0, 50));
                } else if (message.type === 'stats') {
                    // Update stats
                    setStats(message.data);
                } else if (message.type === 'status') {
                    setIsMonitoring(message.monitoring || false);
                }
            };

            ws.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
            };

            ws.onclose = () => {
                console.log('🔌 WebSocket disconnected');
                setIsConnected(false);
                setIsMonitoring(false);

                // Attempt to reconnect after 3 seconds
                reconnectTimeoutRef.current = setTimeout(() => {
                    console.log('🔄 Attempting to reconnect...');
                    connect();
                }, 3000);
            };

            wsRef.current = ws;
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
        }
    }, []);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }

        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        setIsConnected(false);
        setIsMonitoring(false);
    }, []);

    const startMonitoring = useCallback(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: 'start' }));
        }
    }, []);

    const stopMonitoring = useCallback(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: 'stop' }));
        }
    }, []);

    useEffect(() => {
        connect();

        return () => {
            disconnect();
        };
    }, [connect, disconnect]);

    return {
        transactions,
        stats,
        isConnected,
        isMonitoring,
        startMonitoring,
        stopMonitoring,
    };
}
