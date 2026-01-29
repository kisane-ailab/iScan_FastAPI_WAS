import { create } from 'zustand';
import { dockerApi } from '@/shared/api';
import type { Container, ControlAction } from '@/shared/types';

interface ContainerState {
  // State
  containers: Container[];
  isLoading: boolean;
  error: string | null;
  selectedContainer: string | null;
  updatingContainer: string | null; // 업데이트 중인 컨테이너

  // Actions
  fetchContainers: () => Promise<void>;
  controlContainer: (name: string, action: ControlAction) => Promise<void>;
  setSelectedContainer: (name: string | null) => void;
  setUpdatingContainer: (name: string | null) => void;
  clearError: () => void;
}

export const useContainerStore = create<ContainerState>((set, get) => ({
  // Initial state
  containers: [],
  isLoading: false,
  error: null,
  selectedContainer: null,
  updatingContainer: null,

  // Fetch containers
  fetchContainers: async () => {
    set({ isLoading: true, error: null });
    try {
      const containers = await dockerApi.getContainers();
      set({ containers, isLoading: false });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to fetch containers', isLoading: false });
    }
  },

  // Control container
  controlContainer: async (name: string, action: ControlAction) => {
    set({ error: null });
    try {
      await dockerApi.control(name, action);
      // Refresh container list
      await get().fetchContainers();
    } catch (err) {
      set({ error: err instanceof Error ? err.message : `Failed to ${action} container` });
      throw err;
    }
  },

  // Set selected container
  setSelectedContainer: (name) => set({ selectedContainer: name }),

  // Set updating container
  setUpdatingContainer: (name) => set({ updatingContainer: name }),

  // Clear error
  clearError: () => set({ error: null }),
}));
