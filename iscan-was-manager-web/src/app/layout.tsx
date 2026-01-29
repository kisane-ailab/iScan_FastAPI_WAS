import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'iScan WAS Manager',
  description: 'Docker container management for iScan WAS',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
