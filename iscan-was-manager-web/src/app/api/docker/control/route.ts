import { NextRequest, NextResponse } from 'next/server';
import { executeSSHCommand } from '@/shared/lib';
import type { ControlAction } from '@/shared/types';

interface ControlRequest {
  action: ControlAction;
  containerName: string;
  tail?: number;
}

interface ControlResponse {
  success: boolean;
  message: string;
  logs?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: ControlRequest = await request.json();
    const { action, containerName, tail = 100 } = body;

    if (!action || !containerName) {
      return NextResponse.json({
        success: false,
        error: 'Missing required fields: action, containerName',
      }, { status: 400 });
    }

    const validActions: ControlAction[] = ['start', 'stop', 'restart', 'remove', 'logs'];
    if (!validActions.includes(action)) {
      return NextResponse.json({
        success: false,
        error: `Invalid action. Must be one of: ${validActions.join(', ')}`,
      }, { status: 400 });
    }

    let command: string;

    switch (action) {
      case 'start':
        command = `docker start ${containerName}`;
        break;
      case 'stop':
        command = `docker stop ${containerName}`;
        break;
      case 'restart':
        command = `docker restart ${containerName}`;
        break;
      case 'remove':
        command = `docker stop ${containerName} 2>/dev/null || true && docker rm ${containerName}`;
        break;
      case 'logs':
        command = `docker logs --tail ${tail} ${containerName}`;
        break;
      default:
        command = '';
    }

    const result = await executeSSHCommand(command);

    if (result.code !== 0 && action !== 'logs') {
      return NextResponse.json({
        success: false,
        data: {
          success: false,
          message: result.stderr || 'Command failed',
        },
      }, { status: 500 });
    }

    const response: ControlResponse = {
      success: true,
      message: `Action '${action}' completed successfully for ${containerName}`,
    };

    if (action === 'logs') {
      response.logs = result.stdout || result.stderr;
    }

    return NextResponse.json({
      success: true,
      data: response,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({
      success: false,
      error: errorMessage,
    }, { status: 500 });
  }
}
