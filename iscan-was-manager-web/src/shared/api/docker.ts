import { api } from './axios';
import type { Container, ControlAction, UpdateRequest, BrowseResponse } from '../types';

export const dockerApi = {
  // Get container list
  getContainers: async (): Promise<Container[]> => {
    const { data } = await api.get<{ success: boolean; data: Container[]; error?: string }>('/docker/containers');
    if (!data.success) throw new Error(data.error);
    return data.data;
  },

  // Control container (start/stop/restart/remove)
  control: async (containerName: string, action: ControlAction): Promise<string> => {
    const { data } = await api.post<{ success: boolean; data: { message: string }; error?: string }>('/docker/control', {
      containerName,
      action,
    });
    if (!data.success) throw new Error(data.error);
    return data.data.message;
  },

  // Get container logs
  getLogs: async (containerName: string, tail: number = 100): Promise<string> => {
    const { data } = await api.post<{ success: boolean; data: { logs: string }; error?: string }>('/docker/control', {
      containerName,
      action: 'logs',
      tail,
    });
    if (!data.success) throw new Error(data.error);
    return data.data.logs || '';
  },

  // Update container code
  update: async (request: UpdateRequest): Promise<{ message: string; steps: Array<{ step: string; command: string; success: boolean; output: string }> }> => {
    const { data } = await api.post<{ success: boolean; data: { message: string; steps: Array<{ step: string; command: string; success: boolean; output: string }> }; error?: string }>('/docker/update', request);
    if (!data.success) throw new Error(data.error || data.data?.message);
    return data.data;
  },

  // Browse container files
  browse: async (containerName: string, path: string): Promise<BrowseResponse> => {
    const { data } = await api.get<{ success: boolean; data: BrowseResponse; error?: string }>('/docker/browse', {
      params: { container: containerName, path },
    });
    if (!data.success) throw new Error(data.error);
    return data.data;
  },
};
