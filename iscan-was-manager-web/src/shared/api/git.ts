import { api } from './axios';
import type { Branch } from '../types';

export const gitApi = {
  // Get branch list
  getBranches: async (): Promise<Branch[]> => {
    const { data } = await api.get<{ success: boolean; data: Branch[]; error?: string }>('/git/branches');
    if (!data.success) throw new Error(data.error);
    return data.data;
  },
};
