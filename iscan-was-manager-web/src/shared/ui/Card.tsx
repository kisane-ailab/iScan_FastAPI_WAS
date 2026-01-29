'use client';

import { ReactNode } from 'react';

interface CardProps {
  title?: string;
  children: ReactNode;
  action?: ReactNode;
}

export function Card({ title, children, action }: CardProps) {
  return (
    <div
      style={{
        backgroundColor: '#ffffff',
        borderRadius: '12px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06)',
        border: '1px solid #e5e7eb',
        overflow: 'hidden',
      }}
    >
      {title && (
        <div
          style={{
            padding: '18px 20px',
            borderBottom: '1px solid #f3f4f6',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            backgroundColor: '#fafafa',
          }}
        >
          <h2 style={{ fontSize: '16px', fontWeight: '600', color: '#111827', margin: 0 }}>
            {title}
          </h2>
          {action}
        </div>
      )}
      {children}
    </div>
  );
}

export function CardBody({ children, padding = '20px' }: { children: ReactNode; padding?: string }) {
  return <div style={{ padding }}>{children}</div>;
}
