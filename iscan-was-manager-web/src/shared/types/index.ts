// Container types
export type ContainerState = 'running' | 'exited' | 'paused' | 'created' | 'restarting' | 'dead';

export interface Container {
  id: string;
  name: string;
  image: string;
  status: string;
  state: ContainerState;
  ports: string;
  created: string;
}

export type ControlAction = 'start' | 'stop' | 'restart' | 'remove' | 'logs';

export interface UpdateRequest {
  containerName: string;
  branch: string;
}

// Branch types
export interface Branch {
  name: string;
  isRemote: boolean;
  isCurrent: boolean;
}

// File browser types
export interface FileItem {
  name: string;
  type: 'file' | 'directory';
  size?: string;
}

export interface BrowseResponse {
  path: string;
  items: FileItem[];
}
