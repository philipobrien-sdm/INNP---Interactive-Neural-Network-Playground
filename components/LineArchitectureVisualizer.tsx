
import React from 'react';
import { FFNNModel } from '../types';
import { TrainStepResult } from '../types';

// FIX: Added props interface and used React.FC to correctly type component props and handle the 'key' prop.
interface NeuronProps {
    x: number;
    y: number;
    label: string;
    activation: number;
    isSpecial: boolean;
}

const Neuron: React.FC<NeuronProps> = ({ x, y, label, activation, isSpecial }) => {
  const intensity = Math.min(1, Math.abs(activation) * 2.5);
  const color = activation > 0 ? 'rgba(6, 182, 212, 1)' : 'rgba(244, 63, 94, 1)'; // cyan-500, rose-500
  const radius = isSpecial ? 20 : 12;
  return (
    <g transform={`translate(${x}, ${y})`}>
      <circle
        r={radius}
        fill={color}
        style={{ fillOpacity: intensity }}
        stroke="#e5e7eb" // gray-200
        strokeWidth="2"
      />
      <text
        textAnchor="middle"
        dy=".3em"
        className="fill-white font-mono font-bold mix-blend-difference"
        fontSize={isSpecial ? '14px' : '10px'}
      >
        {label}
      </text>
    </g>
  );
};

// FIX: Added props interface and used React.FC to correctly type component props and handle the 'key' prop.
interface ConnectionProps {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    weight: number;
}

const Connection: React.FC<ConnectionProps> = ({ x1, y1, x2, y2, weight }) => {
  const absMaxWeight = 2; // Assume a reasonable max weight for visualization
  const intensity = Math.min(1, Math.abs(weight) / absMaxWeight);
  const color = weight > 0 ? 'rgba(6, 182, 212, 1)' : 'rgba(244, 63, 94, 1)';
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke={color}
      strokeWidth={intensity * 3 + 0.5}
      style={{ strokeOpacity: intensity * 0.8 + 0.1 }}
    />
  );
};

interface LineArchitectureVisualizerProps {
    model: FFNNModel | null;
    visData: TrainStepResult | null;
}

export const LineArchitectureVisualizer: React.FC<LineArchitectureVisualizerProps> = ({ model, visData }) => {
  if (!model || !visData || model.type !== 'FFNN') {
    return (
      <div className="w-full h-full bg-gray-800 rounded-lg flex items-center justify-center text-gray-400 min-h-[400px]">
        Run a step to see visualization.
      </div>
    );
  }

  const { inputToken, activations } = visData;
  const { vocab, tokenToIndex, hiddenLayer, outputLayer } = model;
  
  const inputIndex = tokenToIndex[inputToken];
  const hiddenSize = hiddenLayer.weights.cols;
  const vocabSize = vocab.length;

  const SVG_WIDTH = 500;
  const SVG_HEIGHT = 400;
  const LAYER_X = { input: 50, hidden: 250, output: 450 };

  const getNodeY = (index, total) => (SVG_HEIGHT / (total + 1)) * (index + 1);

  return (
    <div className="bg-gray-800 p-2 rounded-lg border border-gray-700">
      <svg viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`} className="w-full h-auto">
        {/* Connections */}
        {/* Input -> Hidden */}
        {inputIndex !== undefined && Array.from({ length: hiddenSize }).map((_, hIdx) => (
          <Connection
            key={`i-h-${hIdx}`}
            x1={LAYER_X.input}
            y1={getNodeY(inputIndex, vocabSize)}
            x2={LAYER_X.hidden}
            y2={getNodeY(hIdx, hiddenSize)}
            weight={hiddenLayer.weights.data[inputIndex][hIdx]}
          />
        ))}

        {/* Hidden -> Output */}
        {Array.from({ length: hiddenSize }).map((_, hIdx) =>
          Array.from({ length: vocabSize }).map((_, oIdx) => (
            <Connection
              key={`h-o-${hIdx}-${oIdx}`}
              x1={LAYER_X.hidden}
              y1={getNodeY(hIdx, hiddenSize)}
              x2={LAYER_X.output}
              y2={getNodeY(oIdx, vocabSize)}
              weight={outputLayer.weights.data[hIdx][oIdx]}
            />
          ))
        )}

        {/* Neurons */}
        {/* Input Layer */}
        {vocab.map((token, i) => (
          <Neuron
            key={`in-${i}`}
            x={LAYER_X.input}
            y={getNodeY(i, vocabSize)}
            label={token}
            activation={i === inputIndex ? 1 : 0}
            isSpecial={true}
          />
        ))}

        {/* Hidden Layer */}
        {Array.from({ length: hiddenSize }).map((_, i) => (
          <Neuron
            key={`hid-${i}`}
            x={LAYER_X.hidden}
            y={getNodeY(i, hiddenSize)}
            label={`H${i}`}
            activation={activations.hidden.data[0][i]}
            isSpecial={false}
          />
        ))}

        {/* Output Layer */}
        {vocab.map((token, i) => (
          <Neuron
            key={`out-${i}`}
            x={LAYER_X.output}
            y={getNodeY(i, vocabSize)}
            label={token}
            activation={activations.output.data[0][i]}
            isSpecial={true}
          />
        ))}
      </svg>
    </div>
  );
};
