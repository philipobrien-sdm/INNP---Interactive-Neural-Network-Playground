import React, { useState } from 'react';
import { LanguageModel, Matrix } from '../types';
import { Tooltip } from './Tooltip';

interface MatrixHeatmapProps {
  matrix: Matrix;
  title: string;
  tooltipText: string;
}

const MatrixHeatmap: React.FC<MatrixHeatmapProps> = ({ matrix, title, tooltipText }) => {
  if (!matrix || !matrix.data || matrix.rows === 0 || matrix.cols === 0) {
    return null;
  }

  const { rows, cols, data } = matrix;
  const SVG_SIZE = 150;
  const PADDING = 5;
  const cellWidth = (SVG_SIZE - 2 * PADDING) / cols;
  const cellHeight = (SVG_SIZE - 2 * PADDING) / rows;

  let min = Infinity;
  let max = -Infinity;
  data.forEach(row => row.forEach(val => {
    if (val < min) min = val;
    if (val > max) max = val;
  }));

  const absMax = Math.max(Math.abs(min), Math.abs(max));

  const getColor = (value: number) => {
    if (absMax === 0) return 'rgb(107, 114, 128)'; // gray-500
    const intensity = Math.min(1, Math.abs(value) / absMax);
    if (value > 0) {
      // Cyan for positive
      return `rgba(6, 182, 212, ${intensity})`; // cyan-500
    } else {
      // Rose for negative
      return `rgba(244, 63, 94, ${intensity})`; // rose-500
    }
  };

  return (
    <div className="text-center flex flex-col items-center">
      <Tooltip text={tooltipText}>
        <h4 className="text-sm font-semibold text-gray-300 mb-1">{title}</h4>
      </Tooltip>
      <p className="text-xs text-gray-500 mb-2 font-mono">{rows}x{cols}</p>
      <svg width={SVG_SIZE} height={SVG_SIZE} viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`} className="bg-gray-900 rounded-md border border-gray-700">
        {data.map((row, i) =>
          row.map((val, j) => (
            <rect
              key={`${i}-${j}`}
              x={PADDING + j * cellWidth}
              y={PADDING + i * cellHeight}
              width={cellWidth}
              height={cellHeight}
              fill={getColor(val)}
              strokeWidth="0"
            >
              <title>({i},{j}): {val?.toFixed(4) ?? 'N/A'}</title>
            </rect>
          ))
        )}
      </svg>
    </div>
  );
};

const TokenDisplay = ({ label, token, borderColorClass }) => (
  <div className="flex flex-col items-center text-center">
    <h4 className="text-sm font-semibold text-gray-400 mb-1">{label}</h4>
    <div className={`w-16 h-16 flex items-center justify-center bg-gray-900 rounded-md border-2 ${borderColorClass}`}>
      <span className="text-4xl font-mono text-white">{token || '?'}</span>
    </div>
  </div>
);

const MatrixLightbox = ({ matrix, title, onClose }) => {
    if (!matrix) return null;

    const { rows, cols, data } = matrix;
    const MAX_CELLS_TO_RENDER = 5000;
    const isTooLarge = rows * cols > MAX_CELLS_TO_RENDER;

    const getFontSize = () => {
        if (cols > 40) return 'text-[6px]';
        if (cols > 30) return 'text-[8px]';
        if (cols > 20) return 'text-[10px]';
        return 'text-xs';
    };

    return (
        <div 
            className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4"
            onClick={onClose}
        >
            <div 
                className="bg-gray-800 p-6 rounded-lg border border-cyan-500 max-w-6xl max-h-[90vh] overflow-auto relative shadow-2xl"
                onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside the box
            >
                <button 
                    onClick={onClose} 
                    className="absolute top-3 right-3 text-gray-400 hover:text-white"
                    aria-label="Close"
                >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                </button>
                <h2 className="text-2xl font-bold text-cyan-400 mb-4">{title} <span className="text-gray-400 font-mono text-lg">({rows}x{cols})</span></h2>
                
                {isTooLarge ? (
                    <p className="mt-4 text-yellow-400">Matrix is too large ({rows * cols} cells) to display individual values.</p>
                ) : (
                    <div 
                        className="grid gap-px bg-gray-700"
                        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
                    >
                        {data.flat().map((val, index) => (
                            <div key={index} className={`p-1 bg-gray-900 text-center font-mono ${getFontSize()} ${val > 0 ? 'text-cyan-300' : 'text-rose-400'}`}>
                                {val.toFixed(2)}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};


interface ArchitectureVisualizerProps {
  model: LanguageModel | null;
  visData: any;
  isFastMode: boolean;
  currentEpoch: number;
}

export const ArchitectureVisualizer: React.FC<ArchitectureVisualizerProps> = ({ model, visData, isFastMode, currentEpoch }) => {
  const [lightboxMatrix, setLightboxMatrix] = useState<{ matrix: Matrix; title: string } | null>(null);

  if (!model) {
    return (
      <Tooltip text="Visualizes the model's weight matrices as heatmaps. This will be populated once training starts.">
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 min-h-[300px] flex items-center justify-center">
            <p className="text-gray-400">Model not initialized. Train the model to see the architecture.</p>
          </div>
      </Tooltip>
    );
  }
  
  const { inputToken, predictedToken, targetToken } = visData || {};
  const isCorrect = predictedToken === targetToken;

  let predictedBorderColor = 'border-gray-600';
  if (predictedToken && targetToken) {
    predictedBorderColor = isCorrect ? 'border-green-400' : 'border-yellow-400';
  }


  const renderModelMatrices = () => {
    let matrices: { matrix: Matrix, title: string, tooltipText: string }[] = [];
    switch (model.type) {
      case 'FFNN':
        matrices = [
            { matrix: model.hiddenLayer.weights, title: "Hidden Weights", tooltipText: "Weights connecting the input character to the hidden layer. Shape: (vocab_size, hidden_size)." },
            { matrix: model.hiddenLayer.biases, title: "Hidden Biases", tooltipText: "Biases added to each hidden neuron. Shape: (1, hidden_size)." },
            { matrix: model.outputLayer.weights, title: "Output Weights", tooltipText: "Weights connecting the hidden layer to the output logits. Shape: (hidden_size, vocab_size)." },
            { matrix: model.outputLayer.biases, title: "Output Biases", tooltipText: "Biases added to each output neuron. Shape: (1, vocab_size)." },
        ];
        break;
      case 'RNN':
        matrices = [
            { matrix: model.Wxh.weights, title: "Input-Hidden (Wxh)", tooltipText: "Input-to-Hidden Weights (Wxh): Connects the input character to the hidden state." },
            { matrix: model.Whh.weights, title: "Hidden-Hidden (Whh)", tooltipText: "Hidden-to-Hidden Weights (Whh): The recurrent connection that acts as the model's 'memory'." },
            { matrix: model.Why.weights, title: "Hidden-Output (Why)", tooltipText: "Hidden-to-Output Weights (Why): Connects the hidden state to the final output prediction." },
            { matrix: model.Wxh.biases, title: "Input-Hidden Biases", tooltipText: "Biases for the hidden state calculation." },
            { matrix: model.Why.biases, title: "Output Biases", tooltipText: "Biases for the final output prediction." },
        ];
        break;
      case 'GRU':
        matrices = [
                { matrix: model.Wz.weights, title: "Wz (Update)", tooltipText: "Update Gate Weights (Wz): Processes the input to help decide how much of the previous state to keep." },
                { matrix: model.Uz.weights, title: "Uz (Update Rec.)", tooltipText: "Recurrent Update Gate Weights (Uz): Processes the previous state to help decide how much of it to keep." },
                { matrix: model.Wz.biases, title: "bz (Update Bias)", tooltipText: "Update Gate Bias (bz): A learned value added to the update gate's calculation." },
                { matrix: model.Wr.weights, title: "Wr (Reset)", tooltipText: "Reset Gate Weights (Wr): Processes the input to help decide how much of the previous state to forget." },
                { matrix: model.Ur.weights, title: "Ur (Reset Rec.)", tooltipText: "Recurrent Reset Gate Weights (Ur): Processes the previous state to help decide how much of it to forget." },
                { matrix: model.Wr.biases, title: "br (Reset Bias)", tooltipText: "Reset Gate Bias (br): A learned value added to the reset gate's calculation." },
                { matrix: model.Wh.weights, title: "Wh (Candidate)", tooltipText: "Candidate State Weights (Wh): Processes the input to create a new 'candidate' hidden state." },
                { matrix: model.Uh.weights, title: "Uh (Candidate Rec.)", tooltipText: "Recurrent Candidate Weights (Uh): Processes the 'reset' previous state to create the new candidate state." },
                { matrix: model.Wh.biases, title: "bh (Candidate Bias)", tooltipText: "Candidate State Bias (bh): A learned value added to the candidate state's calculation." },
                { matrix: model.Why.weights, title: "Why (Output)", tooltipText: "Output Weights (Why): Connects the final hidden state to the output logits." },
                { matrix: model.Why.biases, title: "by (Output Bias)", tooltipText: "A learned value added to the final output prediction." },
        ];
        break;
      case 'LSTM':
        matrices = [
                { matrix: model.Wf.weights, title: "Wf (Forget)", tooltipText: "Forget Gate Weights (Wf): Processes the input to decide which information to discard from the cell state." },
                { matrix: model.Uf.weights, title: "Uf (Forget Rec.)", tooltipText: "Recurrent Forget Gate Weights (Uf): Processes the previous hidden state for the forget gate." },
                { matrix: model.Wi.weights, title: "Wi (Input)", tooltipText: "Input Gate Weights (Wi): Processes the input to decide which new information to store in the cell state." },
                { matrix: model.Ui.weights, title: "Ui (Input Rec.)", tooltipText: "Recurrent Input Gate Weights (Ui): Processes the previous hidden state for the input gate." },
                { matrix: model.Wo.weights, title: "Wo (Output)", tooltipText: "Output Gate Weights (Wo): Processes the input to decide what to output from the cell state." },
                { matrix: model.Uo.weights, title: "Uo (Output Rec.)", tooltipText: "Recurrent Output Gate Weights (Uo): Processes the previous hidden state for the output gate." },
                { matrix: model.Wc.weights, title: "Wc (Cell)", tooltipText: "Cell State Weights (Wc): Processes the input to create the new candidate cell state." },
                { matrix: model.Uc.weights, title: "Uc (Cell Rec.)", tooltipText: "Recurrent Cell State Weights (Uc): Processes the previous hidden state for the candidate cell state." },
                { matrix: model.Why.weights, title: "Why (Output)", tooltipText: "Output Weights (Why): Connects the final hidden state to the output logits." },
        ];
        break;
      default:
        return <p>Unknown model type.</p>;
    }
    
    return matrices.map(({ matrix, title, tooltipText }) => (
        <div key={title} className="cursor-pointer transform hover:scale-105 transition-transform" onClick={() => setLightboxMatrix({ matrix, title })}>
            <MatrixHeatmap matrix={matrix} title={title} tooltipText={tooltipText} />
        </div>
    ));
  };

  return (
    <>
      <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 min-h-[300px]">
        <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-cyan-400">
                {model.type} Architecture Weights {isFastMode && <span className="text-yellow-400 text-sm">(Updates Suspended)</span>}
            </h3>
            <div className="font-mono text-sm">
                Epoch: <span className="text-gray-300">{currentEpoch}</span>
            </div>
        </div>

        {visData && !isFastMode && (
          <div className="mb-6 pb-4 border-b border-gray-700">
            <h4 className="text-md font-semibold text-gray-300 mb-4 text-center">Current Training Step</h4>
            <div className="flex justify-around items-center">
              <TokenDisplay label="Input" token={inputToken} borderColorClass="border-gray-500" />
              <div className="text-4xl text-gray-600 self-end pb-4">→</div>
              <TokenDisplay label="Target" token={targetToken} borderColorClass="border-green-400" />
              <div className="text-4xl text-gray-600 self-end pb-4">=</div>
              <TokenDisplay label="Predicted" token={predictedToken} borderColorClass={predictedBorderColor} />
            </div>
          </div>
        )}

        <div className="flex flex-wrap gap-4 justify-center items-start">
          {renderModelMatrices()}
        </div>
      </div>
      {lightboxMatrix && (
        <MatrixLightbox
            matrix={lightboxMatrix.matrix}
            title={lightboxMatrix.title}
            onClose={() => setLightboxMatrix(null)}
        />
      )}
    </>
  );
};