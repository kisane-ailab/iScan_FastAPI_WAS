'use client';

interface SpinnerProps {
  size?: number;
  color?: string;
}

export function Spinner({ size = 24, color = '#3b82f6' }: SpinnerProps) {
  return (
    <>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <span
        style={{
          display: 'inline-block',
          width: size,
          height: size,
          border: `3px solid #e5e7eb`,
          borderTopColor: color,
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
        }}
      />
    </>
  );
}
