'use client';

import { Modal, Spinner, Button } from '@/shared/ui';
import { useHostBrowseStore } from '../model/store';

export function HostFileBrowser() {
  const {
    isOpen,
    currentPath,
    items,
    isLoading,
    error,
    closeBrowser,
    navigateTo,
    goUp,
    refresh,
  } = useHostBrowseStore();

  if (!isOpen) return null;

  const pathParts = currentPath.split('/').filter(Boolean);

  const handleItemClick = (item: { name: string; type: string }) => {
    if (item.type === 'directory' || item.type === 'symlink') {
      const newPath = currentPath === '/' ? `/${item.name}` : `${currentPath}/${item.name}`;
      navigateTo(newPath);
    }
  };

  const getIcon = (type: string) => {
    if (type === 'directory') {
      return (
        <svg style={{ width: '20px', height: '20px', color: '#f59e0b' }} fill="currentColor" viewBox="0 0 24 24">
          <path d="M10 4H4a2 2 0 00-2 2v12a2 2 0 002 2h16a2 2 0 002-2V8a2 2 0 00-2-2h-8l-2-2z" />
        </svg>
      );
    }
    if (type === 'symlink') {
      return (
        <svg style={{ width: '20px', height: '20px', color: '#8b5cf6' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
        </svg>
      );
    }
    return (
      <svg style={{ width: '20px', height: '20px', color: '#6b7280' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    );
  };

  const renderBreadcrumb = () => (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '4px',
        padding: '12px 0',
        fontSize: '14px',
        flexWrap: 'wrap',
        borderBottom: '1px solid #e5e7eb',
        marginBottom: '12px',
      }}
    >
      <Button variant="ghost" size="sm" onClick={() => navigateTo('/')}>
        /
      </Button>
      {pathParts.map((part, i) => (
        <span key={i} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: '#9ca3af' }}>/</span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigateTo('/' + pathParts.slice(0, i + 1).join('/'))}
          >
            {part}
          </Button>
        </span>
      ))}
      <div style={{ marginLeft: 'auto' }}>
        <Button variant="ghost" size="sm" onClick={refresh}>
          <svg style={{ width: '16px', height: '16px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </Button>
      </div>
    </div>
  );

  return (
    <Modal title="Host File Browser (192.168.5.103)" onClose={closeBrowser} width="750px">
      {renderBreadcrumb()}

      {isLoading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '48px' }}>
          <Spinner size={32} />
        </div>
      ) : error ? (
        <div style={{ color: '#dc2626', padding: '20px', textAlign: 'center' }}>{error}</div>
      ) : items.length === 0 ? (
        <div style={{ color: '#6b7280', padding: '48px', textAlign: 'center' }}>Empty directory</div>
      ) : (
        <div style={{ maxHeight: '55vh', overflow: 'auto' }}>
          {currentPath !== '/' && (
            <div
              onClick={goUp}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '12px 16px',
                cursor: 'pointer',
                borderBottom: '1px solid #f3f4f6',
                transition: 'background-color 0.15s',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
            >
              <svg style={{ width: '20px', height: '20px', color: '#6b7280' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
              </svg>
              <span style={{ color: '#111827' }}>..</span>
            </div>
          )}
          {items.map((item) => (
            <div
              key={item.name}
              onClick={() => handleItemClick(item)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '12px 16px',
                cursor: item.type === 'directory' || item.type === 'symlink' ? 'pointer' : 'default',
                borderBottom: '1px solid #f3f4f6',
                transition: 'background-color 0.15s',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f9fafb')}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
            >
              {getIcon(item.type)}
              <span style={{ flex: 1, color: '#111827' }}>{item.name}</span>
              <span style={{ fontSize: '12px', color: '#9ca3af', fontFamily: 'monospace' }}>{item.permissions}</span>
              <span style={{ fontSize: '12px', color: '#6b7280', minWidth: '60px', textAlign: 'right' }}>{item.size}</span>
            </div>
          ))}
        </div>
      )}
    </Modal>
  );
}
