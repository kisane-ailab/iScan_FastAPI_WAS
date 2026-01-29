import { NextRequest, NextResponse } from 'next/server';
import { executeSSHCommand } from '@/shared/lib';
import type { FileItem, BrowseResponse } from '@/shared/types';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const containerName = searchParams.get('container');
    const path = searchParams.get('path') || '/';

    if (!containerName) {
      return NextResponse.json({
        success: false,
        error: 'Missing required parameter: container',
      }, { status: 400 });
    }

    const command = `docker exec ${containerName} ls -la "${path}" 2>&1`;
    const result = await executeSSHCommand(command);

    if (result.code !== 0) {
      return NextResponse.json({
        success: false,
        error: result.stderr || result.stdout || 'Failed to list directory',
      }, { status: 500 });
    }

    const lines = result.stdout.trim().split('\n');
    const items: FileItem[] = [];

    for (const line of lines) {
      if (line.startsWith('total') || !line.trim()) continue;

      const parts = line.split(/\s+/);
      if (parts.length < 9) continue;

      const permissions = parts[0];
      const size = parts[4];
      const name = parts.slice(8).join(' ');

      if (name === '.' || name === '..') continue;

      const isDirectory = permissions.startsWith('d');

      items.push({
        name,
        type: isDirectory ? 'directory' : 'file',
        size: isDirectory ? '-' : size,
      });
    }

    items.sort((a, b) => {
      if (a.type === 'directory' && b.type === 'file') return -1;
      if (a.type === 'file' && b.type === 'directory') return 1;
      return a.name.localeCompare(b.name);
    });

    return NextResponse.json({
      success: true,
      data: { path, items } as BrowseResponse,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({
      success: false,
      error: errorMessage,
    }, { status: 500 });
  }
}
