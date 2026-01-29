'use client';

import { useState } from 'react';
import { Button } from '@/shared/ui';
import { useContainerStore } from '@/entities/container';
import type { Container, ControlAction } from '@/shared/types';

interface ContainerActionsProps {
  container: Container;
  onViewLogs: () => void;
  onBrowse: () => void;
}

export function ContainerActions({ container, onViewLogs, onBrowse }: ContainerActionsProps) {
  const [loadingAction, setLoadingAction] = useState<ControlAction | null>(null);
  const { controlContainer } = useContainerStore();

  const handleAction = async (action: ControlAction) => {
    if (action === 'remove' && !confirm(`${container.name} 컨테이너를 삭제하시겠습니까?`)) {
      return;
    }

    setLoadingAction(action);
    try {
      await controlContainer(container.name, action);
    } catch {
      // Error handled in store
    } finally {
      setLoadingAction(null);
    }
  };

  const isLoading = loadingAction !== null;

  return (
    <div style={{ display: 'flex', gap: '6px', justifyContent: 'flex-end', flexWrap: 'wrap' }}>
      {container.state !== 'running' ? (
        <Button
          variant="success"
          size="sm"
          loading={loadingAction === 'start'}
          disabled={isLoading}
          onClick={() => handleAction('start')}
        >
          Start
        </Button>
      ) : (
        <>
          <Button
            variant="warning"
            size="sm"
            loading={loadingAction === 'stop'}
            disabled={isLoading}
            onClick={() => handleAction('stop')}
          >
            Stop
          </Button>
          <Button
            variant="primary"
            size="sm"
            loading={loadingAction === 'restart'}
            disabled={isLoading}
            onClick={() => handleAction('restart')}
          >
            Restart
          </Button>
        </>
      )}
      <Button variant="secondary" size="sm" onClick={onViewLogs}>
        Logs
      </Button>
      <Button
        variant="secondary"
        size="sm"
        onClick={onBrowse}
        style={{ backgroundColor: '#ede9fe', color: '#6d28d9', borderColor: '#ddd6fe' }}
      >
        Browse
      </Button>
      <Button
        variant="danger"
        size="sm"
        loading={loadingAction === 'remove'}
        disabled={isLoading}
        onClick={() => handleAction('remove')}
      >
        Remove
      </Button>
    </div>
  );
}
