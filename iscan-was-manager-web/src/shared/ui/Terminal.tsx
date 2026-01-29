'use client';

import { useEffect, useRef } from 'react';

interface TerminalLine {
  type: 'command' | 'output' | 'success' | 'error' | 'info';
  text: string;
}

interface TerminalProps {
  lines: TerminalLine[];
  title?: string;
  isRunning?: boolean;
}

const styles = {
  container: {
    backgroundColor: '#0d1117',
    borderRadius: '8px',
    overflow: 'hidden',
    fontFamily: "'Consolas', 'Monaco', 'Courier New', monospace",
    fontSize: '13px',
    border: '1px solid #30363d',
  } as React.CSSProperties,
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 14px',
    backgroundColor: '#161b22',
    borderBottom: '1px solid #30363d',
  } as React.CSSProperties,
  dots: {
    display: 'flex',
    gap: '6px',
  } as React.CSSProperties,
  dot: (color: string) => ({
    width: '12px',
    height: '12px',
    borderRadius: '50%',
    backgroundColor: color,
  }) as React.CSSProperties,
  title: {
    flex: 1,
    textAlign: 'center' as const,
    color: '#8b949e',
    fontSize: '12px',
    fontWeight: '500',
  } as React.CSSProperties,
  body: {
    padding: '16px',
    minHeight: '200px',
    maxHeight: '350px',
    overflow: 'auto',
  } as React.CSSProperties,
  line: {
    marginBottom: '8px',
    lineHeight: '1.6',
  } as React.CSSProperties,
  prompt: {
    color: '#7ee787',
    marginRight: '8px',
  } as React.CSSProperties,
  command: {
    color: '#c9d1d9',
  } as React.CSSProperties,
  output: {
    color: '#8b949e',
    paddingLeft: '20px',
    whiteSpace: 'pre-wrap' as const,
    wordBreak: 'break-all' as const,
  } as React.CSSProperties,
  success: {
    color: '#7ee787',
    paddingLeft: '20px',
  } as React.CSSProperties,
  error: {
    color: '#f85149',
    paddingLeft: '20px',
  } as React.CSSProperties,
  info: {
    color: '#58a6ff',
    paddingLeft: '20px',
  } as React.CSSProperties,
  cursor: {
    display: 'inline-block',
    width: '8px',
    height: '16px',
    backgroundColor: '#c9d1d9',
    marginLeft: '4px',
    animation: 'blink 1s step-end infinite',
  } as React.CSSProperties,
};

export function Terminal({ lines, title = 'Terminal', isRunning = false }: TerminalProps) {
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [lines]);

  return (
    <div style={styles.container}>
      <style>{`
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
      `}</style>
      <div style={styles.header}>
        <div style={styles.dots}>
          <div style={styles.dot('#ff5f56')} />
          <div style={styles.dot('#ffbd2e')} />
          <div style={styles.dot('#27c93f')} />
        </div>
        <span style={styles.title}>{title}</span>
        <div style={{ width: '52px' }} />
      </div>
      <div style={styles.body} ref={bodyRef}>
        {lines.length === 0 ? (
          <div style={{ color: '#8b949e' }}>Waiting for commands...</div>
        ) : (
          lines.map((line, i) => (
            <div key={i} style={styles.line}>
              {line.type === 'command' ? (
                <>
                  <span style={styles.prompt}>❯</span>
                  <span style={styles.command}>{line.text}</span>
                </>
              ) : (
                <div style={styles[line.type]}>{line.text}</div>
              )}
            </div>
          ))
        )}
        {isRunning && (
          <div style={styles.line}>
            <span style={styles.prompt}>❯</span>
            <span style={styles.cursor} />
          </div>
        )}
      </div>
    </div>
  );
}
