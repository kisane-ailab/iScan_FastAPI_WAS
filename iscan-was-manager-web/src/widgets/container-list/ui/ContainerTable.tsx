'use client';

import { useEffect, useState } from 'react';
import { Card, Badge, Spinner, Button } from '@/shared/ui';
import { useContainerStore } from '@/entities/container';
import { ContainerActions, LogViewer } from '@/features/container-control';
import { useBrowseStore, FileBrowser } from '@/features/file-browse';

const styles = {
  centered: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    padding: '60px 20px',
    gap: '12px',
    color: '#6b7280',
  } as React.CSSProperties,
  errorBox: {
    padding: '30px',
    textAlign: 'center' as const,
    color: '#dc2626',
  } as React.CSSProperties,
  emptyBox: {
    padding: '60px 20px',
    textAlign: 'center' as const,
    color: '#6b7280',
    fontSize: '15px',
  } as React.CSSProperties,
  tableWrap: {
    overflowX: 'auto' as const,
  } as React.CSSProperties,
  table: {
    width: '100%',
    borderCollapse: 'collapse' as const,
    fontSize: '14px',
  } as React.CSSProperties,
  th: {
    padding: '14px 16px',
    textAlign: 'left' as const,
    fontSize: '12px',
    fontWeight: '600',
    color: '#6b7280',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
    backgroundColor: '#f9fafb',
    borderBottom: '1px solid #e5e7eb',
  } as React.CSSProperties,
  td: {
    padding: '16px',
    borderBottom: '1px solid #f3f4f6',
    verticalAlign: 'middle' as const,
  } as React.CSSProperties,
  nameWrap: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '2px',
  } as React.CSSProperties,
  name: {
    fontWeight: '600',
    color: '#111827',
    fontSize: '14px',
  } as React.CSSProperties,
  id: {
    fontSize: '12px',
    color: '#9ca3af',
    fontFamily: 'monospace',
  } as React.CSSProperties,
  text: {
    color: '#374151',
    fontSize: '13px',
  } as React.CSSProperties,
};

export function ContainerTable() {
  const { containers, isLoading, error, fetchContainers, updatingContainer } = useContainerStore();
  const { openBrowser } = useBrowseStore();
  const [logsContainer, setLogsContainer] = useState<string | null>(null);

  useEffect(() => {
    fetchContainers();
    const interval = setInterval(fetchContainers, 10000);
    return () => clearInterval(interval);
  }, [fetchContainers]);

  const refreshBtn = (
    <Button variant="ghost" size="sm" onClick={fetchContainers} style={{ padding: '8px' }}>
      <svg style={{ width: '18px', height: '18px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
      </svg>
    </Button>
  );

  if (isLoading && containers.length === 0) {
    return (
      <Card title="Docker Containers" action={refreshBtn}>
        <div style={styles.centered}>
          <Spinner size={24} />
          <span>Loading containers...</span>
        </div>
      </Card>
    );
  }

  if (error && containers.length === 0) {
    return (
      <Card title="Docker Containers" action={refreshBtn}>
        <div style={styles.errorBox}>
          <p style={{ marginBottom: '12px' }}>Error: {error}</p>
          <Button variant="danger" size="sm" onClick={fetchContainers}>
            Retry
          </Button>
        </div>
      </Card>
    );
  }

  return (
    <>
      <Card title="Docker Containers" action={refreshBtn}>
        {containers.length === 0 ? (
          <div style={styles.emptyBox}>No iScanInstance containers found</div>
        ) : (
          <div style={styles.tableWrap}>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>Name</th>
                  <th style={styles.th}>Status</th>
                  <th style={styles.th}>Image</th>
                  <th style={styles.th}>Ports</th>
                  <th style={{ ...styles.th, textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {containers.map((c) => (
                  <tr key={c.id}>
                    <td style={styles.td}>
                      <div style={styles.nameWrap}>
                        <span style={styles.name}>{c.name}</span>
                        <span style={styles.id}>{c.id.slice(0, 12)}</span>
                      </div>
                    </td>
                    <td style={styles.td}>
                      {updatingContainer === c.name ? (
                        <Badge state="updating" />
                      ) : (
                        <Badge state={c.state} />
                      )}
                    </td>
                    <td style={{ ...styles.td, ...styles.text }}>{c.image}</td>
                    <td style={{ ...styles.td, ...styles.text }}>{c.ports || '-'}</td>
                    <td style={{ ...styles.td, textAlign: 'right' }}>
                      <ContainerActions
                        container={c}
                        onViewLogs={() => setLogsContainer(c.name)}
                        onBrowse={() => openBrowser(c.name)}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {logsContainer && <LogViewer containerName={logsContainer} onClose={() => setLogsContainer(null)} />}
      <FileBrowser />
    </>
  );
}
