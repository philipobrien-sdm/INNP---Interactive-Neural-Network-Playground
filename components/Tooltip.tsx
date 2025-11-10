/**
 * @file Tooltip.tsx
 * @description A reusable tooltip component that displays a descriptive text box
 * when the user hovers over its child element. This is used extensively to explain
 * complex concepts and controls throughout the UI.
 */

import React, { ReactNode } from 'react';

/**
 * A higher-order component that wraps its children with a hover-triggered tooltip.
 * It uses CSS `group-hover` utility from Tailwind CSS to control visibility, making it
 * lightweight and performant.
 * @param {object} props - The component's props.
 * @param {string} props.text - The text to display inside the tooltip.
 * @param {string} [props.className] - Optional additional class names for the wrapper.
 * @param {ReactNode} props.children - The element that will trigger the tooltip on hover.
 */
// FIX: Using React.FC to correctly type the component props, including children.
// FIX: Explicitly added children to the props type, as it's no longer implicit in React 18's types for React.FC.
export const Tooltip: React.FC<{ text: string, className?: string, children: ReactNode }> = ({ children, text, className }) => {
  return (
    // The 'group' class allows child elements to change their style based on the parent's state (e.g., hover).
    <div className={`relative group ${className || ''}`}>
      {children}
      {/* 
        The tooltip element itself. It's positioned absolutely relative to the wrapper.
        It starts with `opacity-0` and becomes `opacity-100` when the parent `group` is hovered.
        `pointer-events-none` ensures the tooltip itself doesn't interfere with mouse events.
      */}
      <div className="absolute bottom-full mb-2 w-max max-w-xs p-2 text-xs text-white bg-gray-800 border border-cyan-500 rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-10 pointer-events-none transform -translate-x-1/2 left-1/2" role="tooltip">
        {text}
      </div>
    </div>
  );
};
