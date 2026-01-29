'use client';

import { useState, useEffect, useMemo } from 'react';
import { Card, CardBody, Select, Button, Terminal } from '@/shared/ui';
import { useContainerStore } from '@/entities/container';
import { BranchSelector, useUpdateStore } from '@/features/code-update';

const styles = {
  formGroup: {
    marginBottom: '16px',
  } as React.CSSProperties,
  label: {
    display: 'block',
    fontSize: '13px',
    fontWeight: '600',
    color: '#374151',
    marginBottom: '8px',
  } as React.CSSProperties,
  hint: {
    marginTop: '6px',
    fontSize: '12px',
    color: '#9ca3af',
  } as React.CSSProperties,
  successBox: {
    marginTop: '16px',
    padding: '14px',
    backgroundColor: '#ecfdf5',
    color: '#065f46',
    borderRadius: '8px',
    fontSize: '14px',
    fontWeight: '500',
    border: '1px solid #a7f3d0',
  } as React.CSSProperties,
  errorBox: {
    marginTop: '16px',
    padding: '14px',
    backgroundColor: '#fef2f2',
    color: '#991b1b',
    borderRadius: '8px',
    fontSize: '14px',
    fontWeight: '500',
    border: '1px solid #fecaca',
  } as React.CSSProperties,
  terminalWrap: {
    marginTop: '20px',
  } as React.CSSProperties,
};

interface TerminalLine {
  type: 'command' | 'output' | 'success' | 'error' | 'info';
  text: string;
}

export function UpdatePanel() {
  const [selectedContainer, setSelectedContainer] = useState('');
  const [selectedBranch, setSelectedBranch] = useState('');

  const { containers, fetchContainers, setUpdatingContainer } = useContainerStore();
  const { isUpdating, steps, status, error, updateContainer } = useUpdateStore();

  useEffect(() => {
    if (containers.length === 0) {
      fetchContainers();
    }
  }, [containers.length, fetchContainers]);

  // 업데이트 완료시 컨테이너 새로고침
  useEffect(() => {
    if (status === 'success' || status === 'error') {
      setUpdatingContainer(null);
      fetchContainers();
    }
  }, [status, setUpdatingContainer, fetchContainers]);

  const handleUpdate = async () => {
    if (!selectedContainer) {
      alert('컨테이너를 선택해주세요');
      return;
    }
    if (!selectedBranch) {
      alert('브랜치를 선택해주세요');
      return;
    }

    setUpdatingContainer(selectedContainer);
    await updateContainer(selectedContainer, selectedBranch);
  };

  // Convert steps to terminal lines
  const terminalLines = useMemo<TerminalLine[]>(() => {
    const lines: TerminalLine[] = [];
    for (const step of steps) {
      // Add step name as info
      lines.push({ type: 'info', text: `▸ ${step.step}` });
      // Add command
      lines.push({ type: 'command', text: step.command });
      // Add output
      if (step.output && step.output !== '(no output)') {
        const outputLines = step.output.split('\n').filter(l => l.trim());
        for (const line of outputLines) {
          lines.push({
            type: step.success ? 'output' : 'error',
            text: line
          });
        }
      }
      // Add result
      lines.push({
        type: step.success ? 'success' : 'error',
        text: step.success ? '✓ Done' : '✗ Failed'
      });
    }
    return lines;
  }, [steps]);

  const containerOptions = containers.map((c) => ({
    value: c.name,
    label: `${c.name} (${c.state})`,
  }));

  const isDisabled = isUpdating || !selectedContainer || !selectedBranch;

  return (
    <Card title="Update Code">
      <CardBody padding="16px">
        <div style={styles.formGroup}>
          <label style={styles.label}>Container</label>
          <Select
            value={selectedContainer}
            onChange={setSelectedContainer}
            options={containerOptions}
            placeholder="Select container..."
          />
        </div>

        <div style={styles.formGroup}>
          <label style={styles.label}>Git Branch</label>
          <BranchSelector value={selectedBranch} onChange={setSelectedBranch} />
          <p style={styles.hint}>iscan-was-manager-web 폴더 제외</p>
        </div>

        <Button
          variant="success"
          loading={isUpdating}
          disabled={isDisabled}
          onClick={handleUpdate}
          style={{ width: '100%', padding: '12px 16px' }}
        >
          <svg style={{ width: '18px', height: '18px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Update & Restart
        </Button>

        {status === 'success' && (
          <div style={styles.successBox}>
            ✓ 코드 업데이트 및 컨테이너 재시작 완료!
          </div>
        )}

        {status === 'error' && (
          <div style={styles.errorBox}>
            ✗ 업데이트 실패: {error}
          </div>
        )}

        {(isUpdating || steps.length > 0) && (
          <div style={styles.terminalWrap}>
            <Terminal
              lines={terminalLines}
              title={`${selectedContainer || 'container'} - Update`}
              isRunning={isUpdating}
            />
          </div>
        )}
      </CardBody>
    </Card>
  );
}
