'use client';

import { ButtonHTMLAttributes, ReactNode } from 'react';

type Variant = 'primary' | 'success' | 'warning' | 'danger' | 'secondary' | 'ghost';
type Size = 'sm' | 'md' | 'lg';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  children: ReactNode;
}

const variantStyles: Record<Variant, { bg: string; color: string; border: string; hoverBg: string }> = {
  primary: { bg: '#2563eb', color: '#ffffff', border: '#2563eb', hoverBg: '#1d4ed8' },
  success: { bg: '#059669', color: '#ffffff', border: '#059669', hoverBg: '#047857' },
  warning: { bg: '#d97706', color: '#ffffff', border: '#d97706', hoverBg: '#b45309' },
  danger: { bg: '#dc2626', color: '#ffffff', border: '#dc2626', hoverBg: '#b91c1c' },
  secondary: { bg: '#f3f4f6', color: '#374151', border: '#e5e7eb', hoverBg: '#e5e7eb' },
  ghost: { bg: 'transparent', color: '#6b7280', border: 'transparent', hoverBg: '#f3f4f6' },
};

const sizeStyles: Record<Size, { padding: string; fontSize: string; borderRadius: string }> = {
  sm: { padding: '6px 12px', fontSize: '12px', borderRadius: '6px' },
  md: { padding: '10px 16px', fontSize: '14px', borderRadius: '8px' },
  lg: { padding: '12px 20px', fontSize: '15px', borderRadius: '8px' },
};

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled,
  children,
  style,
  ...props
}: ButtonProps) {
  const v = variantStyles[variant];
  const s = sizeStyles[size];
  const isDisabled = disabled || loading;

  return (
    <button
      disabled={isDisabled}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '8px',
        border: `1px solid ${v.border}`,
        borderRadius: s.borderRadius,
        padding: s.padding,
        fontSize: s.fontSize,
        fontWeight: '500',
        backgroundColor: v.bg,
        color: v.color,
        cursor: isDisabled ? 'not-allowed' : 'pointer',
        opacity: isDisabled ? 0.6 : 1,
        transition: 'background-color 0.15s, opacity 0.15s',
        outline: 'none',
        ...style,
      }}
      onMouseEnter={(e) => {
        if (!isDisabled) {
          (e.target as HTMLButtonElement).style.backgroundColor = v.hoverBg;
        }
      }}
      onMouseLeave={(e) => {
        (e.target as HTMLButtonElement).style.backgroundColor = v.bg;
      }}
      {...props}
    >
      {loading && (
        <span
          style={{
            width: '14px',
            height: '14px',
            border: '2px solid rgba(255,255,255,0.3)',
            borderTopColor: 'currentColor',
            borderRadius: '50%',
            animation: 'spin 0.8s linear infinite',
            flexShrink: 0,
          }}
        />
      )}
      {children}
    </button>
  );
}
