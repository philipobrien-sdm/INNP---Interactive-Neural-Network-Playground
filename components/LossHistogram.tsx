
import React from 'react';

export const LossHistogram = ({ lossHistory }) => {
  if (!lossHistory || lossHistory.length === 0) {
    return (
      <div className="flex-grow bg-gray-800 p-4 rounded-lg border border-gray-700 flex flex-col items-center justify-center">
        <h2 className="text-xl font-semibold text-cyan-400">Loss Per Epoch</h2>
        <p className="text-gray-400 mt-4">Loss history will be displayed here after training.</p>
      </div>
    );
  }

  const SVG_WIDTH = 500;
  const SVG_HEIGHT = 250;
  const PADDING = { top: 20, right: 10, bottom: 30, left: 45 };
  const CHART_WIDTH = SVG_WIDTH - PADDING.left - PADDING.right;
  const CHART_HEIGHT = SVG_HEIGHT - PADDING.top - PADDING.bottom;

  const maxLoss = Math.max(...lossHistory, 0) * 1.1; // Add 10% ceiling
  const barWidth = CHART_WIDTH / lossHistory.length;

  const yTicks = 5;
  const tickValues = Array.from({ length: yTicks + 1 }, (_, i) => (maxLoss / yTicks) * i);

  return (
    <div className="flex-grow bg-gray-800 p-4 rounded-lg border border-gray-700 flex flex-col overflow-hidden">
      <div className="flex-shrink-0">
        <h2 className="text-xl font-semibold mb-2 text-cyan-400">Loss Per Epoch</h2>
        <p className="text-sm text-gray-400 mb-4">
          Average training loss at the end of each epoch. Lower is better.
        </p>
      </div>
      <div className="flex-grow flex items-center justify-center">
        <svg viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`} className="w-full h-full" aria-label="Histogram of training loss per epoch">
          {/* Y-axis with labels */}
          <line x1={PADDING.left} y1={PADDING.top} x2={PADDING.left} y2={PADDING.top + CHART_HEIGHT} className="stroke-gray-600" />
          {tickValues.map((value, i) => {
            const y = PADDING.top + CHART_HEIGHT - (value / maxLoss) * CHART_HEIGHT;
            return (
              <g key={i}>
                <line x1={PADDING.left - 5} y1={y} x2={PADDING.left} y2={y} className="stroke-gray-600" />
                <text x={PADDING.left - 8} y={y + 4} textAnchor="end" className="fill-gray-400 text-[10px]">
                  {value.toFixed(2)}
                </text>
              </g>
            );
          })}
          <text transform={`translate(${PADDING.left/3}, ${SVG_HEIGHT/2}) rotate(-90)`} textAnchor="middle" className="fill-gray-400 text-xs">
            Avg Loss
          </text>

          {/* X-axis */}
          <line x1={PADDING.left} y1={PADDING.top + CHART_HEIGHT} x2={PADDING.left + CHART_WIDTH} y2={PADDING.top + CHART_HEIGHT} className="stroke-gray-600" />
           <text x={SVG_WIDTH / 2} y={SVG_HEIGHT - 5} textAnchor="middle" className="fill-gray-400 text-xs">
            Epoch
          </text>

          {/* Bars */}
          {lossHistory.map((loss, index) => {
            const barHeight = loss > 0 ? (loss / maxLoss) * CHART_HEIGHT : 0;
            const x = PADDING.left + index * barWidth;
            const y = PADDING.top + CHART_HEIGHT - barHeight;
            return (
              <rect
                key={index}
                x={x}
                y={y}
                width={Math.max(1, barWidth - 2)} // Ensure bar is visible, add gap
                height={barHeight}
                className="fill-cyan-600 hover:fill-cyan-400 transition-colors"
              >
                <title>Epoch {index + 1}: Loss {loss.toFixed(4)}</title>
              </rect>
            );
          })}
        </svg>
      </div>
    </div>
  );
};