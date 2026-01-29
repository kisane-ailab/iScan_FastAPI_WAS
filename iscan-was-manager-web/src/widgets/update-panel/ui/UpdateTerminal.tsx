'use client';

import { useMemo } from 'react';
import { Terminal } from '@/shared/ui';
import { useUpdateStore } from '@/features/code-update';
import { useContainerStore } from '@/entities/container';

interface TerminalLine {
  type: 'command' | 'output' | 'success' | 'error' | 'info';
  text: string;
}

export function UpdateTerminal() {
  const { isUpdating, steps } = useUpdateStore();
  const { updatingContainer } = useContainerStore();

  const terminalLines = useMemo<TerminalLine[]>(() => {
    const lines: TerminalLine[] = [];
    for (const step of steps) {
      lines.push({ type: 'info', text: `▸ ${step.step}` });
      lines.push({ type: 'command', text: step.command });
      if (step.output && step.output !== '(no output)') {
        const outputLines = step.output.split('\n').filter(l => l.trim());
        for (const line of outputLines) {
          lines.push({
            type: step.success ? 'output' : 'error',
            text: line
          });
        }
      }
      lines.push({
        type: step.success ? 'success' : 'error',
        text: step.success ? '✓ Done' : '✗ Failed'
      });
    }
    return lines;
  }, [steps]);

  return (
    <Terminal
      lines={terminalLines}
      title={updatingContainer ? `${updatingContainer} - Update` : 'Output'}
      isRunning={isUpdating}
    />
  );
}
