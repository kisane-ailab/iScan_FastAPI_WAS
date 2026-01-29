import { create } from 'zustand';
import { dockerApi } from '@/shared/api';
import type { FileItem } from '@/shared/types';

interface BrowseState {
  containerName: string | null;
  currentPath: string;
  items: FileItem[];
  isLoading: boolean;
  error: string | null;

  openBrowser: (containerName: string) => void;
  closeBrowser: () => void;
  navigateTo: (path: string) => Promise<void>;
  goUp: () => Promise<void>;
  refresh: () => Promise<void>;
}

export const useBrowseStore = create<BrowseState>((set, get) => ({
  containerName: null,
  currentPath: '/',
  items: [],
  isLoading: false,
  error: null,

  openBrowser: (containerName: string) => {
    set({ containerName, currentPath: '/', items: [], error: null });
    get().navigateTo('/');
  },

  closeBrowser: () => {
    set({ containerName: null, currentPath: '/', items: [], error: null });
  },

  navigateTo: async (path: string) => {
    const { containerName } = get();
    if (!containerName) return;

    set({ isLoading: true, error: null, currentPath: path });
    try {
      const data = await dockerApi.browse(containerName, path);
      set({ items: data.items, isLoading: false });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to browse', isLoading: false });
    }
  },

  goUp: async () => {
    const { currentPath } = get();
    if (currentPath === '/') return;

    const parts = currentPath.split('/').filter(Boolean);
    parts.pop();
    const newPath = parts.length === 0 ? '/' : '/' + parts.join('/');
    await get().navigateTo(newPath);
  },

  refresh: async () => {
    const { currentPath } = get();
    await get().navigateTo(currentPath);
  },
}));
