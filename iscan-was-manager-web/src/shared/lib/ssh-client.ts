import { Client, ConnectConfig } from 'ssh2';

export interface SSHConfig {
  host: string;
  port: number;
  username: string;
  password: string;
}

export interface CommandResult {
  stdout: string;
  stderr: string;
  code: number;
}

export function getSSHConfig(): SSHConfig {
  return {
    host: process.env.SSH_HOST || '',
    port: parseInt(process.env.SSH_PORT || '22', 10),
    username: process.env.SSH_USERNAME || '',
    password: process.env.SSH_PASSWORD || '',
  };
}

export async function executeSSHCommand(
  command: string,
  config?: SSHConfig
): Promise<CommandResult> {
  const sshConfig = config || getSSHConfig();

  return new Promise((resolve, reject) => {
    const conn = new Client();

    conn.on('ready', () => {
      conn.exec(command, (err, stream) => {
        if (err) {
          conn.end();
          return reject(err);
        }

        let stdout = '';
        let stderr = '';

        stream.on('close', (code: number) => {
          conn.end();
          resolve({ stdout, stderr, code: code || 0 });
        });

        stream.on('data', (data: Buffer) => {
          stdout += data.toString();
        });

        stream.stderr.on('data', (data: Buffer) => {
          stderr += data.toString();
        });
      });
    });

    conn.on('error', (err) => {
      reject(err);
    });

    const connectConfig: ConnectConfig = {
      host: sshConfig.host,
      port: sshConfig.port,
      username: sshConfig.username,
      password: sshConfig.password,
      readyTimeout: 10000,
    };

    conn.connect(connectConfig);
  });
}
