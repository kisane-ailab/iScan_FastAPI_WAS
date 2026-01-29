import { NextResponse } from 'next/server';
import { executeSSHCommand } from '@/shared/lib';

export async function GET() {
  try {
    // Test 1: docker ps with ports
    const ps = await executeSSHCommand(`docker ps -a --filter "name=iScanInstance" --format "{{.Names}}|{{.Ports}}"`);

    // Test 2: docker inspect for first container
    const firstContainer = await executeSSHCommand(`docker ps -a --filter "name=iScanInstance" --format "{{.Names}}" | head -1`);
    const containerName = firstContainer.stdout.trim();

    let inspect = null;
    if (containerName) {
      const inspectResult = await executeSSHCommand(`docker inspect ${containerName} --format '{{json .NetworkSettings.Ports}}'`);
      inspect = {
        containerName,
        raw: inspectResult.stdout,
        parsed: inspectResult.stdout.trim(),
      };
    }

    // Test 3: docker port command
    let portCommand = null;
    if (containerName) {
      const portResult = await executeSSHCommand(`docker port ${containerName}`);
      portCommand = portResult.stdout || portResult.stderr || '(empty)';
    }

    return NextResponse.json({
      success: true,
      data: {
        dockerPs: ps.stdout,
        inspect,
        portCommand,
      },
    });
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
