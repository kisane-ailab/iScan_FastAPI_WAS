import { create } from 'zustand';
import { gitApi } from '@/shared/api';
import type { Branch } from '@/shared/types';

interface BranchState {
  // State
  branches: Branch[];
  isLoading: boolean;
  error: string | null;
  selectedBranch: string;

  // Actions
  fetchBranches: () => Promise<void>;
  setSelectedBranch: (branch: string) => void;
  clearError: () => void;
}

export const useBranchStore = create<BranchState>((set) => ({
  // Initial state
  branches: [],
  isLoading: false,
  error: null,
  selectedBranch: '',

  // Fetch branches
  fetchBranches: async () => {
    set({ isLoading: true, error: null });
    try {
      const branches = await gitApi.getBranches();
      set({ branches, isLoading: false });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to fetch branches', isLoading: false });
    }
  },

  // Set selected branch
  setSelectedBranch: (branch) => set({ selectedBranch: branch }),

  // Clear error
  clearError: () => set({ error: null }),
}));
