import { NextRequest } from 'next/server';
import { executeSSHCommand } from '@/shared/lib';
import type { UpdateRequest } from '@/shared/types';

interface StepResult {
  step: string;
  command: string;
  success: boolean;
  output: string;
}

async function runStep(step: string, command: string): Promise<StepResult> {
  const result = await executeSSHCommand(command);
  return {
    step,
    command,
    success: result.code === 0,
    output: result.stdout + (result.stderr ? `\n[stderr] ${result.stderr}` : '') || '(no output)',
  };
}

export async function POST(request: NextRequest) {
  const body: UpdateRequest = await request.json();
  const { containerName, branch } = body;

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      const sendStep = (step: StepResult, done: boolean = false, error: string | null = null) => {
        const data = JSON.stringify({ step, done, error });
        controller.enqueue(encoder.encode(`data: ${data}\n\n`));
      };

      try {
        if (!containerName || !branch) {
          sendStep({ step: 'Error', command: '-', success: false, output: 'Missing containerName or branch' }, true, 'Missing required fields');
          controller.close();
          return;
        }

        const repoPath = process.env.GIT_REPO_PATH || '/home/iscaninstance/iScan_FastAPI_WAS';

        // Step 1: Check if repo exists
        const checkStep = await runStep(
          'Check Repository',
          `test -d ${repoPath}/.git && echo "Repository exists at ${repoPath}" || echo "Repository NOT found at ${repoPath}"`
        );
        sendStep(checkStep);

        const repoExists = checkStep.output.includes('exists');
        if (!repoExists) {
          // Clone the repo
          const gitUrl = `https://${process.env.GITHUB_PAT}@github.com/${process.env.GITHUB_OWNER}/${process.env.GITHUB_REPO}.git`;
          const cloneStep = await runStep(
            'Git Clone',
            `git clone ${gitUrl} ${repoPath} 2>&1`
          );
          sendStep(cloneStep);
          if (!cloneStep.success) {
            sendStep(cloneStep, true, 'Clone failed');
            controller.close();
            return;
          }
        }

        // Step 2: Git fetch
        const fetchStep = await runStep(
          'Git Fetch',
          `cd ${repoPath} && git fetch --all 2>&1`
        );
        sendStep(fetchStep);
        if (!fetchStep.success) {
          sendStep(fetchStep, true, 'Fetch failed');
          controller.close();
          return;
        }

        // Step 3: Git checkout
        const checkoutStep = await runStep(
          'Git Checkout',
          `cd ${repoPath} && git checkout ${branch} 2>&1`
        );
        sendStep(checkoutStep);
        if (!checkoutStep.success) {
          sendStep(checkoutStep, true, 'Checkout failed');
          controller.close();
          return;
        }

        // Step 4: Git pull
        const pullStep = await runStep(
          'Git Pull',
          `cd ${repoPath} && git pull origin ${branch} 2>&1`
        );
        sendStep(pullStep);
        if (!pullStep.success) {
          sendStep(pullStep, true, 'Pull failed');
          controller.close();
          return;
        }

        // Step 5: Docker cp
        const cpStep = await runStep(
          'Docker Copy',
          `docker cp ${repoPath}/. ${containerName}:/iScan_FastAPI_WAS/ 2>&1 || docker cp ${repoPath}/. ${containerName}:/app/ 2>&1`
        );
        sendStep(cpStep);
        if (!cpStep.success) {
          sendStep(cpStep, true, 'Docker copy failed');
          controller.close();
          return;
        }

        // Step 6: Remove web manager folder
        const rmStep = await runStep(
          'Remove Web Manager',
          `docker exec ${containerName} rm -rf /iScan_FastAPI_WAS/iscan-was-manager-web 2>&1; docker exec ${containerName} rm -rf /app/iscan-was-manager-web 2>&1; echo "cleanup done"`
        );
        sendStep(rmStep);

        // Step 7: Docker restart
        const restartStep = await runStep(
          'Docker Restart',
          `docker restart ${containerName} 2>&1`
        );
        sendStep(restartStep);
        if (!restartStep.success) {
          sendStep(restartStep, true, 'Restart failed');
          controller.close();
          return;
        }

        // Done
        sendStep({ step: 'Complete', command: '-', success: true, output: `Container ${containerName} updated and restarted` }, true, null);
        controller.close();

      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        sendStep({ step: 'Error', command: '-', success: false, output: errorMessage }, true, errorMessage);
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}
