'use client';

import { ContainerTable } from '@/widgets/container-list';
import { UpdateForm, UpdateTerminal } from '@/widgets/update-panel';
import { ServerStatus } from '@/widgets/server-status';
import { HostFileBrowser, useHostBrowseStore } from '@/features/host-browse';
import { Button } from '@/shared/ui';

const styles = {
  wrapper: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column' as const,
  },
  header: {
    backgroundColor: '#ffffff',
    borderBottom: '1px solid #e5e7eb',
    padding: '20px 0',
  },
  headerInner: {
    maxWidth: '1600px',
    margin: '0 auto',
    padding: '0 24px',
  },
  headerTop: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  logo: {
    width: '36px',
    height: '36px',
    color: '#2563eb',
  },
  title: {
    fontSize: '22px',
    fontWeight: '700',
    color: '#111827',
    margin: 0,
  },
  subtitle: {
    marginTop: '6px',
    fontSize: '14px',
    color: '#6b7280',
  },
  main: {
    flex: 1,
    maxWidth: '1600px',
    width: '100%',
    margin: '0 auto',
    padding: '24px',
  },
  topSection: {
    marginBottom: '24px',
  },
  bottomSection: {
    display: 'flex',
    gap: '24px',
    alignItems: 'flex-start',
  },
  formCol: {
    width: '320px',
    flexShrink: 0,
  },
  terminalCol: {
    flex: 1,
    minWidth: 0,
  },
  footer: {
    padding: '20px 24px',
    textAlign: 'center' as const,
    fontSize: '13px',
    color: '#9ca3af',
    borderTop: '1px solid #e5e7eb',
    backgroundColor: '#ffffff',
  },
};

export default function HomePage() {
  const { openBrowser } = useHostBrowseStore();

  return (
    <div style={styles.wrapper}>
      <header style={styles.header}>
        <div style={styles.headerInner}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={styles.headerTop}>
              <svg style={styles.logo} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
              </svg>
              <h1 style={styles.title}>iScan WAS Manager</h1>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={openBrowser}
              style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <svg style={{ width: '16px', height: '16px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
              </svg>
              Host Files
            </Button>
          </div>
          <p style={styles.subtitle}>Docker container management for 192.168.5.103</p>
        </div>
      </header>

      <ServerStatus />

      <main style={styles.main}>
        <div style={styles.topSection}>
          <ContainerTable />
        </div>

        <div style={styles.bottomSection}>
          <div style={styles.formCol}>
            <UpdateForm />
          </div>
          <div style={styles.terminalCol}>
            <UpdateTerminal />
          </div>
        </div>
      </main>

      <footer style={styles.footer}>
        iScan WAS Manager &copy; Kisan Inc.
      </footer>

      <HostFileBrowser />
    </div>
  );
}
