'use client';

import { create } from 'zustand';
import { api } from '@/shared/api';

interface FileItem {
  name: string;
  type: 'file' | 'directory' | 'symlink';
  size: string;
  permissions: string;
}

interface HostBrowseState {
  isOpen: boolean;
  currentPath: string;
  items: FileItem[];
  isLoading: boolean;
  error: string | null;

  openBrowser: () => void;
  closeBrowser: () => void;
  navigateTo: (path: string) => Promise<void>;
  goUp: () => void;
  refresh: () => void;
}

export const useHostBrowseStore = create<HostBrowseState>((set, get) => ({
  isOpen: false,
  currentPath: '/home/iscaninstance',
  items: [],
  isLoading: false,
  error: null,

  openBrowser: () => {
    set({ isOpen: true });
    get().navigateTo('/home/iscaninstance');
  },

  closeBrowser: () => {
    set({ isOpen: false, items: [], currentPath: '/home/iscaninstance', error: null });
  },

  navigateTo: async (path: string) => {
    set({ isLoading: true, error: null });
    try {
      const res = await api.get('/host/browse', { params: { path } });
      if (res.data.success) {
        set({
          currentPath: res.data.data.path,
          items: res.data.data.items,
          isLoading: false,
        });
      } else {
        set({ error: res.data.error, isLoading: false });
      }
    } catch (err) {
      set({
        error: err instanceof Error ? err.message : 'Failed to browse',
        isLoading: false,
      });
    }
  },

  goUp: () => {
    const { currentPath, navigateTo } = get();
    const parts = currentPath.split('/').filter(Boolean);
    if (parts.length > 0) {
      parts.pop();
      navigateTo('/' + parts.join('/') || '/');
    }
  },

  refresh: () => {
    const { currentPath, navigateTo } = get();
    navigateTo(currentPath);
  },
}));
