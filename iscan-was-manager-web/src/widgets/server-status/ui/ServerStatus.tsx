'use client';

import { useEffect, useState } from 'react';
import { api } from '@/shared/api';

interface StatusData {
  hostname: string;
  uptime: string;
  cpu: {
    usage: number;
    cores: number;
  };
  memory: {
    total: string;
    used: string;
    free: string;
    usagePercent: number;
  };
  disk: {
    total: string;
    used: string;
    free: string;
    usagePercent: number;
  };
}

const styles = {
  container: {
    display: 'flex',
    gap: '24px',
    alignItems: 'center',
    padding: '12px 20px',
    backgroundColor: '#f8fafc',
    borderBottom: '1px solid #e5e7eb',
    fontSize: '13px',
  } as React.CSSProperties,
  item: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  } as React.CSSProperties,
  label: {
    color: '#6b7280',
    fontWeight: '500',
  } as React.CSSProperties,
  value: {
    color: '#111827',
    fontWeight: '600',
  } as React.CSSProperties,
  bar: {
    width: '60px',
    height: '6px',
    backgroundColor: '#e5e7eb',
    borderRadius: '3px',
    overflow: 'hidden',
  } as React.CSSProperties,
  barFill: (percent: number, color: string) => ({
    width: `${percent}%`,
    height: '100%',
    backgroundColor: color,
    borderRadius: '3px',
    transition: 'width 0.3s ease',
  }) as React.CSSProperties,
  hostname: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    color: '#111827',
    fontWeight: '600',
  } as React.CSSProperties,
  uptime: {
    color: '#6b7280',
    fontSize: '12px',
  } as React.CSSProperties,
  error: {
    color: '#dc2626',
    fontSize: '12px',
  } as React.CSSProperties,
};

function getBarColor(percent: number): string {
  if (percent >= 90) return '#ef4444';
  if (percent >= 70) return '#f59e0b';
  return '#10b981';
}

export function ServerStatus() {
  const [status, setStatus] = useState<StatusData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchStatus = async (showLoading = false) => {
    if (showLoading) setIsRefreshing(true);
    try {
      const res = await api.get('/host/status');
      if (res.data.success) {
        setStatus(res.data.data);
        setError(null);
      } else {
        setError(res.data.error);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch');
    } finally {
      if (showLoading) setIsRefreshing(false);
    }
  };

  const handleRefresh = () => {
    fetchStatus(true);
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // 30초마다 갱신
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div style={styles.container}>
        <span style={styles.error}>Server status unavailable: {error}</span>
      </div>
    );
  }

  if (!status) {
    return (
      <div style={styles.container}>
        <span style={styles.label}>Loading server status...</span>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.hostname}>
        <svg style={{ width: '16px', height: '16px', color: '#10b981' }} fill="currentColor" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="10" />
        </svg>
        {status.hostname}
        <span style={styles.uptime}>({status.uptime})</span>
      </div>

      <div style={styles.item}>
        <span style={styles.label}>CPU:</span>
        <div style={styles.bar}>
          <div style={styles.barFill(status.cpu.usage, getBarColor(status.cpu.usage))} />
        </div>
        <span style={styles.value}>{status.cpu.usage.toFixed(1)}%</span>
        <span style={{ color: '#9ca3af', fontSize: '11px' }}>({status.cpu.cores} cores)</span>
      </div>

      <div style={styles.item}>
        <span style={styles.label}>Memory:</span>
        <div style={styles.bar}>
          <div style={styles.barFill(status.memory.usagePercent, getBarColor(status.memory.usagePercent))} />
        </div>
        <span style={styles.value}>{status.memory.usagePercent.toFixed(1)}%</span>
        <span style={{ color: '#9ca3af', fontSize: '11px' }}>({status.memory.used}/{status.memory.total})</span>
      </div>

      <div style={styles.item}>
        <span style={styles.label}>Disk:</span>
        <div style={styles.bar}>
          <div style={styles.barFill(status.disk.usagePercent, getBarColor(status.disk.usagePercent))} />
        </div>
        <span style={styles.value}>{status.disk.usagePercent}%</span>
        <span style={{ color: '#9ca3af', fontSize: '11px' }}>({status.disk.used}/{status.disk.total})</span>
      </div>

      <button
        onClick={handleRefresh}
        disabled={isRefreshing}
        style={{
          marginLeft: 'auto',
          padding: '6px',
          background: 'none',
          border: 'none',
          cursor: isRefreshing ? 'not-allowed' : 'pointer',
          color: '#6b7280',
          display: 'flex',
          alignItems: 'center',
          borderRadius: '4px',
          transition: 'background-color 0.15s',
        }}
        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#e5e7eb')}
        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
      >
        <svg
          style={{
            width: '16px',
            height: '16px',
            animation: isRefreshing ? 'spin 1s linear infinite' : 'none',
          }}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </button>
    </div>
  );
}
