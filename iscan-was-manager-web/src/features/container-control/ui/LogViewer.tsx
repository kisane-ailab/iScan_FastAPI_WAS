'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Modal, Spinner } from '@/shared/ui';
import { dockerApi } from '@/shared/api';

interface LogViewerProps {
  containerName: string;
  onClose: () => void;
}

export function LogViewer({ containerName, onClose }: LogViewerProps) {
  const [logs, setLogs] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLPreElement>(null);

  const fetchLogs = useCallback(async () => {
    try {
      const data = await dockerApi.getLogs(containerName, 200);
      setLogs(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch logs');
    } finally {
      setIsLoading(false);
    }
  }, [containerName]);

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [fetchLogs]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <Modal
      title={`Logs: ${containerName}`}
      onClose={onClose}
      width="900px"
      footer={<span style={{ fontSize: '12px', color: '#6b7280' }}>5초마다 자동 새로고침</span>}
    >
      {isLoading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '48px' }}>
          <Spinner size={32} />
        </div>
      ) : error ? (
        <div style={{ color: '#dc2626', padding: '20px' }}>Error: {error}</div>
      ) : (
        <pre
          ref={logRef}
          style={{
            backgroundColor: '#1f2937',
            color: '#f3f4f6',
            padding: '16px',
            borderRadius: '6px',
            minHeight: '400px',
            maxHeight: '60vh',
            overflow: 'auto',
            fontSize: '13px',
            fontFamily: 'Consolas, Monaco, monospace',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
            lineHeight: '1.5',
            margin: 0,
          }}
        >
          {logs || 'No logs available'}
        </pre>
      )}
    </Modal>
  );
}
