import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  // Mark ssh2 as external for server-side only
  serverExternalPackages: ['ssh2'],
  // Use webpack instead of Turbopack for builds
  bundlePagesRouterDependencies: false,
};

export default nextConfig;
