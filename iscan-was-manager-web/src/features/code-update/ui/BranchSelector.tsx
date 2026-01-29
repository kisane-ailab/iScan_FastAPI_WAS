'use client';

import { useEffect } from 'react';
import { Select, Spinner, Button } from '@/shared/ui';
import { useBranchStore } from '@/entities/branch';

interface BranchSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export function BranchSelector({ value, onChange }: BranchSelectorProps) {
  const { branches, isLoading, error, fetchBranches } = useBranchStore();

  useEffect(() => {
    if (branches.length === 0) {
      fetchBranches();
    }
  }, [branches.length, fetchBranches]);

  if (isLoading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Spinner size={16} />
        <span style={{ fontSize: '14px', color: '#6b7280' }}>Loading branches...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ fontSize: '14px', color: '#dc2626' }}>Error loading branches</span>
        <Button variant="ghost" size="sm" onClick={fetchBranches}>
          Retry
        </Button>
      </div>
    );
  }

  const options = branches.map((b) => ({
    value: b.name,
    label: b.isCurrent ? `${b.name} (current)` : b.name,
  }));

  return (
    <div style={{ display: 'flex', gap: '8px' }}>
      <div style={{ flex: 1 }}>
        <Select
          value={value}
          onChange={onChange}
          options={options}
          placeholder="Select branch..."
        />
      </div>
      <Button variant="ghost" size="sm" onClick={fetchBranches} style={{ padding: '10px' }}>
        <svg style={{ width: '18px', height: '18px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </Button>
    </div>
  );
}
