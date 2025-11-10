
import React, { useState, useEffect, useMemo } from 'react';

interface PredictionStats {
  [fromToken: string]: {
    [toToken: string]: {
      correct: number;
      total: number;
    };
  };
}

interface SuccessRateHeatmapProps {
  statsHistory: PredictionStats[];
  vocab: string[];
}

export const SuccessRateHeatmap: React.FC<SuccessRateHeatmapProps> = ({ statsHistory, vocab }) => {
  const [displayEpoch, setDisplayEpoch] = useState(1);

  useEffect(() => {
    // When new data comes in, jump to the latest epoch
    if (statsHistory.length > 0) {
      setDisplayEpoch(statsHistory.length);
    }
  }, [statsHistory.length]);

  const cumulativeStats = useMemo(() => {
    if (!statsHistory || statsHistory.length === 0) {
      return {};
    }

    const relevantHistory = statsHistory.slice(0, displayEpoch);
    const aggregatedStats: PredictionStats = {};

    relevantHistory.forEach((epochStats, epochIndex) => {
      const weight = epochIndex + 1; // Linear weight for the epoch

      for (const fromToken in epochStats) {
        if (!aggregatedStats[fromToken]) {
          aggregatedStats[fromToken] = {};
        }
        for (const toToken in epochStats[fromToken]) {
          if (!aggregatedStats[fromToken][toToken]) {
            aggregatedStats[fromToken][toToken] = { correct: 0, total: 0 };
          }
          const epochPairStats = epochStats[fromToken][toToken];
          // Apply the weight to this epoch's stats before aggregating
          aggregatedStats[fromToken][toToken].correct += epochPairStats.correct * weight;
          aggregatedStats[fromToken][toToken].total += epochPairStats.total * weight;
        }
      }
    });

    return aggregatedStats;
  }, [statsHistory, displayEpoch]);

  if (!statsHistory || statsHistory.length === 0 || !vocab || vocab.length === 0) {
    return (
      <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 flex flex-col items-center justify-center min-h-[300px]">
        <h2 className="text-xl font-semibold text-cyan-400">Prediction Success Rate Timeline</h2>
        <p className="text-gray-400 mt-4">Prediction statistics will appear here once training begins.</p>
      </div>
    );
  }

  // Filter out whitespace tokens for a cleaner heatmap
  const filteredVocab = vocab.filter(t => t.trim() !== '');
  const vocabSize = filteredVocab.length;
  if (vocabSize === 0) {
     return (
        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 flex flex-col items-center justify-center min-h-[300px]">
            <h2 className="text-xl font-semibold text-cyan-400">Prediction Success Rate Timeline</h2>
            <p className="text-gray-400 mt-4">No vocabulary to display.</p>
        </div>
     );
  }
  
  const stats = cumulativeStats;

  const SVG_WIDTH = 500;
  const SVG_HEIGHT = 500;
  
  const PADDING_WITH_LABELS = { top: 50, right: 10, bottom: 10, left: 50 };
  const PADDING_WITHOUT_LABELS = { top: 10, right: 10, bottom: 10, left: 10 };

  const tempCellSize = Math.min(
    (SVG_WIDTH - PADDING_WITH_LABELS.left - PADDING_WITH_LABELS.right) / vocabSize,
    (SVG_HEIGHT - PADDING_WITH_LABELS.top - PADDING_WITH_LABELS.bottom) / vocabSize
  );
  
  const showLabels = tempCellSize >= 10;
  const PADDING = showLabels ? PADDING_WITH_LABELS : PADDING_WITHOUT_LABELS;

  const CELL_SIZE = Math.min(
    (SVG_WIDTH - PADDING.left - PADDING.right) / vocabSize,
    (SVG_HEIGHT - PADDING.top - PADDING.bottom) / vocabSize
  );
  
  const FONT_SIZE = showLabels ? Math.max(6, Math.min(12, CELL_SIZE * 0.6)) : 0;

  const getColor = (rate: number | undefined) => {
    if (rate === undefined || isNaN(rate)) return 'rgb(55 65 81)'; // gray-700 for untried pairs
    // From red (0%) to yellow (50%) to green (100%)
    const hue = rate * 120;
    return `hsl(${hue}, 100%, 45%)`;
  };
  
  return (
    <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 flex flex-col">
       <h2 className="text-xl font-semibold mb-2 text-cyan-400">Prediction Success Rate Timeline</h2>
        <p className="text-sm text-gray-400 mb-4">
          Heatmap showing the model's <strong className="text-gray-200">weighted cumulative accuracy</strong> for predicting the next token (columns) given an input token (rows). Later epochs are given more importance in the calculation. Use the slider to see progress over time. {!showLabels && <span className="text-yellow-400">Labels hidden for large vocabularies.</span>}
        </p>
      <div className="flex-grow flex items-center justify-center overflow-auto">
        <svg viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`} style={{ width: '100%', height: 'auto' }}>
          {showLabels && (
            <>
              {/* Column Headers (Target) */}
              {filteredVocab.map((token, i) => (
                <text
                    key={`col-header-${i}`}
                    x={PADDING.left + i * CELL_SIZE + CELL_SIZE / 2}
                    y={PADDING.top - 8}
                    textAnchor="middle"
                    fontSize={FONT_SIZE}
                    className="fill-gray-400 font-mono"
                  >
                  {token === ' ' ? "' '" : token}
                </text>
              ))}
              {/* Row Headers (Input) */}
              {filteredVocab.map((token, i) => (
                <text
                  key={`row-header-${i}`}
                  x={PADDING.left - 8}
                  y={PADDING.top + i * CELL_SIZE + CELL_SIZE / 2}
                  textAnchor="end"
                  dy=".3em"
                  fontSize={FONT_SIZE}
                  className="fill-gray-400 font-mono"
                >
                  {token === ' ' ? "' '" : token}
                </text>
              ))}
              
              <text 
                transform={`translate(${PADDING.left + (CELL_SIZE * vocabSize) / 2}, 20)`} 
                textAnchor="middle" 
                className="fill-gray-300 text-sm font-semibold">
                Target Token
              </text>
              <text 
                transform={`translate(20, ${PADDING.top + (CELL_SIZE * vocabSize) / 2}) rotate(-90)`}
                textAnchor="middle" 
                className="fill-gray-300 text-sm font-semibold">
                Input Token
              </text>
            </>
          )}

          {/* Heatmap Cells */}
          {filteredVocab.map((fromToken, i) => 
            filteredVocab.map((toToken, j) => {
              const pairStats = stats[fromToken]?.[toToken];
              const rate = pairStats && pairStats.total > 0 ? pairStats.correct / pairStats.total : undefined;
              return (
                <g key={`${i}-${j}`}>
                  <rect
                    x={PADDING.left + j * CELL_SIZE}
                    y={PADDING.top + i * CELL_SIZE}
                    width={CELL_SIZE}
                    height={CELL_SIZE}
                    fill={getColor(rate)}
                    stroke="rgba(107, 114, 128, 0.2)"
                    strokeWidth="1"
                  >
                    <title>{`${fromToken} → ${toToken}: ${pairStats ? `${pairStats.correct.toFixed(0)}/${pairStats.total.toFixed(0)} (${(rate !== undefined ? rate*100 : 0).toFixed(1)}%)` : '0/0 (N/A)'}`}</title>
                  </rect>
                </g>
              )
            })
          )}
        </svg>
      </div>
      <div className="mt-4 flex items-center space-x-4">
        <label htmlFor="epoch-slider" className="flex-shrink-0 text-sm font-medium text-gray-300">
          Cumulative Epoch: <span className="font-mono bg-gray-900 px-2 py-1 rounded-md">{displayEpoch}</span>
        </label>
        <input
            id="epoch-slider"
            type="range"
            min="1"
            max={statsHistory.length}
            value={displayEpoch}
            onChange={(e) => setDisplayEpoch(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
            disabled={statsHistory.length <= 1}
        />
      </div>
    </div>
  );
};
