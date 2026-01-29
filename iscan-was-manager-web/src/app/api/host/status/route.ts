import { NextResponse } from 'next/server';
import { executeSSHCommand } from '@/shared/lib';

interface ServerStatus {
  hostname: string;
  uptime: string;
  cpu: {
    usage: number;
    cores: number;
  };
  memory: {
    total: string;
    used: string;
    free: string;
    usagePercent: number;
  };
  disk: {
    total: string;
    used: string;
    free: string;
    usagePercent: number;
  };
}

export async function GET() {
  try {
    // Get hostname
    const hostnameResult = await executeSSHCommand('hostname');
    const hostname = hostnameResult.stdout.trim();

    // Get uptime
    const uptimeResult = await executeSSHCommand('uptime -p');
    const uptime = uptimeResult.stdout.trim().replace('up ', '');

    // Get CPU info
    const cpuCoresResult = await executeSSHCommand('nproc');
    const cpuCores = parseInt(cpuCoresResult.stdout.trim(), 10) || 1;

    // Get CPU usage (1 second average)
    const cpuUsageResult = await executeSSHCommand("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'");
    const cpuUsage = parseFloat(cpuUsageResult.stdout.trim()) || 0;

    // Get memory info
    const memResult = await executeSSHCommand("free -h | awk '/^Mem:/ {print $2, $3, $4}'");
    const memParts = memResult.stdout.trim().split(/\s+/);
    const memTotal = memParts[0] || '0';
    const memUsed = memParts[1] || '0';
    const memFree = memParts[2] || '0';

    // Get memory usage percent
    const memPercentResult = await executeSSHCommand("free | awk '/^Mem:/ {printf \"%.1f\", $3/$2 * 100}'");
    const memUsagePercent = parseFloat(memPercentResult.stdout.trim()) || 0;

    // Get disk info for root partition
    const diskResult = await executeSSHCommand("df -h / | awk 'NR==2 {print $2, $3, $4, $5}'");
    const diskParts = diskResult.stdout.trim().split(/\s+/);
    const diskTotal = diskParts[0] || '0';
    const diskUsed = diskParts[1] || '0';
    const diskFree = diskParts[2] || '0';
    const diskUsagePercent = parseInt(diskParts[3]?.replace('%', '') || '0', 10);

    const status: ServerStatus = {
      hostname,
      uptime,
      cpu: {
        usage: cpuUsage,
        cores: cpuCores,
      },
      memory: {
        total: memTotal,
        used: memUsed,
        free: memFree,
        usagePercent: memUsagePercent,
      },
      disk: {
        total: diskTotal,
        used: diskUsed,
        free: diskFree,
        usagePercent: diskUsagePercent,
      },
    };

    return NextResponse.json({
      success: true,
      data: status,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({
      success: false,
      error: errorMessage,
    }, { status: 500 });
  }
}
