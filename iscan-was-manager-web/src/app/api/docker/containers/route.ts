import { NextResponse } from 'next/server';
import { executeSSHCommand } from '@/shared/lib';
import type { Container } from '@/shared/types';

export async function GET() {
  try {
    // Get container list - use docker inspect for more reliable port info
    const listCommand = `docker ps -a --filter "name=iScanInstance" --format "{{.ID}}"`;
    const listResult = await executeSSHCommand(listCommand);

    if (listResult.code !== 0 && listResult.stderr) {
      return NextResponse.json({
        success: false,
        error: listResult.stderr,
      }, { status: 500 });
    }

    const containerIds = listResult.stdout.trim().split('\n').filter(id => id.trim());

    if (containerIds.length === 0) {
      return NextResponse.json({
        success: true,
        data: [],
      });
    }

    // Get detailed info for each container using docker inspect
    const containers: Container[] = [];

    for (const id of containerIds) {
      const inspectCommand = `docker inspect ${id} --format '{{.Id}}|{{.Name}}|{{.Config.Image}}|{{.State.Status}}|{{range $p, $conf := .NetworkSettings.Ports}}{{$p}}->{{range $conf}}{{.HostPort}}{{end}},{{end}}|{{.Created}}'`;
      const inspectResult = await executeSSHCommand(inspectCommand);

      if (inspectResult.code === 0 && inspectResult.stdout.trim()) {
        const line = inspectResult.stdout.trim();
        const parts = line.split('|');
        const fullId = parts[0] || '';
        const name = (parts[1] || '').replace(/^\//, ''); // Remove leading slash
        const image = parts[2] || '';
        const state = parts[3] || 'unknown';
        const portsRaw = parts[4] || '';
        const created = parts[5] || '';

        // Parse ports: "8080/tcp->8080,443/tcp->443," => "8080, 443"
        const ports = portsRaw
          .split(',')
          .filter(p => p.includes('->'))
          .map(p => {
            const match = p.match(/(\d+)\/\w+->(\d+)/);
            if (match) {
              return match[1] === match[2] ? match[2] : `${match[2]}:${match[1]}`;
            }
            return null;
          })
          .filter(Boolean)
          .join(', ');

        containers.push({
          id: fullId.slice(0, 12),
          name,
          image,
          status: state,
          state: state.toLowerCase() as Container['state'],
          ports: ports || '-',
          created,
        });
      }
    }

    return NextResponse.json({
      success: true,
      data: containers,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({
      success: false,
      error: errorMessage,
    }, { status: 500 });
  }
}
