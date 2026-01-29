import { NextRequest, NextResponse } from 'next/server';
import { executeSSHCommand } from '@/shared/lib';

interface BuildRequest {
  branch: string;
  containerName: string;
  port: number;
}

interface BuildResponse {
  success: boolean;
  message: string;
  containerId?: string;
  logs?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: BuildRequest = await request.json();
    const { branch, containerName, port } = body;

    if (!branch || !containerName || !port) {
      return NextResponse.json({
        success: false,
        error: 'Missing required fields: branch, containerName, port',
      }, { status: 400 });
    }

    if (!containerName.startsWith('iScanInstance.')) {
      return NextResponse.json({
        success: false,
        error: 'Container name must start with "iScanInstance."',
      }, { status: 400 });
    }

    const repoPath = process.env.GIT_REPO_PATH || '/home/iscaninstance/iScan_FastAPI_WAS';

    const commands = [
      `cd ${repoPath}`,
      `git fetch --all`,
      `git checkout ${branch}`,
      `git pull origin ${branch}`,
      `docker stop ${containerName} 2>/dev/null || true`,
      `docker rm ${containerName} 2>/dev/null || true`,
      `docker build -t ${containerName.toLowerCase()}:latest .`,
      `docker run -d --gpus all --name ${containerName} -p ${port}:50000 -v /home/iscaninstance/mynas:/mynas ${containerName.toLowerCase()}:latest`,
    ];

    const fullCommand = commands.join(' && ');
    const result = await executeSSHCommand(fullCommand);

    if (result.code !== 0) {
      return NextResponse.json({
        success: false,
        data: {
          success: false,
          message: 'Build failed',
          logs: result.stderr || result.stdout,
        } as BuildResponse,
      }, { status: 500 });
    }

    const containerId = result.stdout.trim().split('\n').pop() || '';

    return NextResponse.json({
      success: true,
      data: {
        success: true,
        message: `Container ${containerName} built and started successfully`,
        containerId,
        logs: result.stdout,
      } as BuildResponse,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({
      success: false,
      error: errorMessage,
    }, { status: 500 });
  }
}
