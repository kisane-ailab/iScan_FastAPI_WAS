import { NextRequest, NextResponse } from 'next/server';
import { executeSSHCommand } from '@/shared/lib';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const path = searchParams.get('path') || '/home/iscaninstance';

    // List files with details
    const command = `ls -la "${path}" 2>/dev/null | tail -n +2`;
    const result = await executeSSHCommand(command);

    if (result.code !== 0) {
      return NextResponse.json({
        success: false,
        error: result.stderr || 'Failed to list directory',
      }, { status: 500 });
    }

    const items = result.stdout
      .split('\n')
      .filter((line) => line.trim())
      .map((line) => {
        const parts = line.split(/\s+/);
        const permissions = parts[0] || '';
        const size = parts[4] || '0';
        const name = parts.slice(8).join(' ');

        if (!name || name === '.' || name === '..') return null;

        const type = permissions.startsWith('d') ? 'directory' :
                     permissions.startsWith('l') ? 'symlink' : 'file';

        return {
          name,
          type,
          size: type === 'directory' ? '-' : formatSize(parseInt(size, 10)),
          permissions,
        };
      })
      .filter(Boolean);

    return NextResponse.json({
      success: true,
      data: {
        path,
        items,
      },
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({
      success: false,
      error: errorMessage,
    }, { status: 500 });
  }
}

function formatSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}
