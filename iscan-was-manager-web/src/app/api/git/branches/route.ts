import { NextResponse } from 'next/server';
import type { Branch } from '@/shared/types';

interface GitHubBranch {
  name: string;
  protected: boolean;
}

export async function GET() {
  try {
    const pat = process.env.GITHUB_PAT;
    const owner = process.env.GITHUB_OWNER;
    const repo = process.env.GITHUB_REPO;

    if (!pat || !owner || !repo) {
      return NextResponse.json({
        success: false,
        error: 'GitHub configuration missing (PAT, OWNER, or REPO)',
      }, { status: 500 });
    }

    const response = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/branches?per_page=100`,
      {
        headers: {
          'Authorization': `Bearer ${pat}`,
          'Accept': 'application/vnd.github+json',
          'X-GitHub-Api-Version': '2022-11-28',
        },
        cache: 'no-store',
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json({
        success: false,
        error: `GitHub API error: ${response.status} - ${errorText}`,
      }, { status: response.status });
    }

    const githubBranches: GitHubBranch[] = await response.json();

    const branches: Branch[] = githubBranches
      .map((branch) => ({
        name: branch.name,
        isRemote: true,
        isCurrent: branch.name === 'main' || branch.name === 'master',
      }))
      .sort((a, b) => {
        if (a.name === 'main' || a.name === 'master') return -1;
        if (b.name === 'main' || b.name === 'master') return 1;
        return a.name.localeCompare(b.name);
      });

    return NextResponse.json({
      success: true,
      data: branches,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({
      success: false,
      error: errorMessage,
    }, { status: 500 });
  }
}
