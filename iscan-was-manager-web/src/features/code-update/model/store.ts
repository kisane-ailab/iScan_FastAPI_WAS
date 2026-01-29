import { create } from 'zustand';

interface StepResult {
  step: string;
  command: string;
  success: boolean;
  output: string;
}

interface UpdateState {
  isUpdating: boolean;
  steps: StepResult[];
  status: 'idle' | 'success' | 'error';
  error: string | null;

  updateContainer: (containerName: string, branch: string) => Promise<void>;
  reset: () => void;
}

export const useUpdateStore = create<UpdateState>((set, get) => ({
  isUpdating: false,
  steps: [],
  status: 'idle',
  error: null,

  updateContainer: async (containerName: string, branch: string) => {
    set({ isUpdating: true, steps: [], status: 'idle', error: null });

    try {
      const response = await fetch('/api/docker/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ containerName, branch }),
      });

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              const { step, done: isDone, error } = data;

              if (step) {
                set((state) => ({
                  steps: [...state.steps, step],
                }));
              }

              if (isDone) {
                if (error) {
                  set({ isUpdating: false, status: 'error', error });
                } else {
                  set({ isUpdating: false, status: 'success', error: null });
                }
              }
            } catch {
              // Ignore JSON parse errors
            }
          }
        }
      }
    } catch (err) {
      set({
        isUpdating: false,
        status: 'error',
        error: err instanceof Error ? err.message : 'Update failed',
      });
    }
  },

  reset: () => set({ isUpdating: false, steps: [], status: 'idle', error: null }),
}));
