'use client';

import type { ContainerState } from '../types';

type BadgeState = ContainerState | 'updating';

interface BadgeProps {
  state: BadgeState;
}

const stateConfig: Record<BadgeState, { bg: string; color: string; border: string; dot: string; animate?: boolean }> = {
  running: { bg: '#dcfce7', color: '#166534', border: '#bbf7d0', dot: '#22c55e' },
  exited: { bg: '#fee2e2', color: '#991b1b', border: '#fecaca', dot: '#ef4444' },
  paused: { bg: '#fef3c7', color: '#92400e', border: '#fde68a', dot: '#f59e0b' },
  created: { bg: '#f3f4f6', color: '#374151', border: '#e5e7eb', dot: '#9ca3af' },
  restarting: { bg: '#dbeafe', color: '#1e40af', border: '#bfdbfe', dot: '#3b82f6' },
  dead: { bg: '#fee2e2', color: '#991b1b', border: '#fecaca', dot: '#ef4444' },
  updating: { bg: '#faf5ff', color: '#7c3aed', border: '#e9d5ff', dot: '#8b5cf6', animate: true },
};

export function Badge({ state }: BadgeProps) {
  const config = stateConfig[state] || stateConfig.created;

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: '4px 10px',
        borderRadius: '9999px',
        fontSize: '12px',
        fontWeight: '500',
        backgroundColor: config.bg,
        color: config.color,
        border: `1px solid ${config.border}`,
      }}
    >
      {config.animate ? (
        <span
          style={{
            width: '14px',
            height: '14px',
            border: '2px solid rgba(139, 92, 246, 0.3)',
            borderTopColor: config.dot,
            borderRadius: '50%',
            animation: 'spin 0.8s linear infinite',
          }}
        />
      ) : (
        <span
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: config.dot,
          }}
        />
      )}
      {state}
    </span>
  );
}
