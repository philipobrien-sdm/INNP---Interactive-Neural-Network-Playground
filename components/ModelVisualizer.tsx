import React from 'react';
import { Tooltip } from './Tooltip';
import { LanguageModel, TrainStepResult } from '../types';

// FIX: Explicitly typed Neuron props and used React.FC to resolve issue with 'key' prop.
interface NeuronProps {
  activation: number;
  label: string;
  isTarget: boolean;
  isPredicted: boolean;
  isInput: boolean;
}

const Neuron: React.FC<NeuronProps> = ({ activation, label, isTarget, isPredicted, isInput }) => {
  // Increased multiplier from 2 to 2.5 for a brighter appearance
  const intensity = isInput ? 1 : Math.min(1, Math.max(0, activation) * 2.5);
  const color = activation > 0 ? 'bg-cyan-500' : 'bg-red-500';
  const sizeClass = isInput || isTarget || isPredicted ? "w-10 h-10 text-xs" : "w-6 h-6";
  
  let borderColor = 'border-gray-600';
  if (isTarget && isPredicted) {
    borderColor = 'border-green-400'; // Correct prediction
  } else if (isTarget) {
    borderColor = 'border-green-400'; // The correct target
  } else if (isPredicted) {
    borderColor = 'border-yellow-400'; // The incorrect prediction
  }
  
  return (
    <div
      className={`relative rounded-full border-2 ${borderColor} flex items-center justify-center transition-all duration-200 ${sizeClass}`}
      style={{ backgroundColor: isInput ? 'rgba(107, 114, 128, 0.5)' : color, opacity: intensity, transition: 'background-color 0.2s, opacity 0.2s' }}
      title={`Activation: ${activation?.toFixed(4) ?? 'N/A'}`}
    >
      <span className="text-white mix-blend-difference font-mono font-semibold">{label}</span>
      {isTarget && !isPredicted && <div className="absolute -top-2 -right-2 w-4 h-4 bg-green-400 rounded-full text-black text-xs flex items-center justify-center font-bold" title="Target">T</div>}
      {isPredicted && !isTarget && <div className="absolute -bottom-2 -right-2 w-4 h-4 bg-yellow-400 rounded-full text-black text-xs flex items-center justify-center font-bold" title="Predicted">P</div>}
      {isPredicted && isTarget && <div className="absolute -top-2 -right-2 w-4 h-4 bg-green-400 rounded-full text-black text-xs flex items-center justify-center font-bold" title="Correct">✓</div>}
    </div>
  );
};

export const ModelVisualizer = ({ model, modelType, visData, status, isFastMode }) => {
  if (!model) return null;

  const hiddenSize = 'hiddenLayer' in model ? model.hiddenLayer.weights.cols : 'Wxh' in model ? model.Wxh.weights.cols : model.Why.weights.rows;
  const hiddenLayerLabels = Array.from({ length: hiddenSize }, (_, i) => `H${i + 1}`);
  const showRecurrent = modelType === 'RNN' || modelType === 'GRU';

  const { inputToken, targetToken, predictedToken, activations } = visData || {};

  const renderOutputNeurons = () => {
    if (!visData) {
        return <div className="w-10 h-10"></div>;
    }

    if (targetToken === predictedToken) {
        if (!targetToken) return <div className="w-10 h-10"></div>;
        const tokenIndex = model.tokenToIndex[targetToken];
        const activation = activations?.output?.data?.[0]?.[tokenIndex] ?? 0;
        return (
            <Neuron
                key={`${targetToken}-correct`}
                label={targetToken}
                activation={activation}
                isInput={false}
                isTarget={true}
                isPredicted={true}
            />
        );
    } else {
        return (
            <>
                {targetToken && (
                    <Neuron
                        key={`${targetToken}-target`}
                        label={targetToken}
                        activation={activations?.output?.data?.[0]?.[model.tokenToIndex[targetToken]] ?? 0}
                        isInput={false}
                        isTarget={true}
                        isPredicted={false}
                    />
                )}
                {predictedToken && (
                    <Neuron
                        key={`${predictedToken}-predicted`}
                        label={predictedToken}
                        activation={activations?.output?.data?.[0]?.[model.tokenToIndex[predictedToken]] ?? 0}
                        isInput={false}
                        isTarget={false}
                        isPredicted={true}
                    />
                )}
            </>
        );
    }
  };

  return (
    <Tooltip text="Visualizes the neuron activations for the current training step. Brighter colors indicate stronger activation. 'T' is the correct target character, 'P' is the model's prediction.">
        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 min-h-[300px]">
        <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-cyan-400">
                {modelType} Architecture {isFastMode && <span className="text-yellow-400 text-sm">(Updates Suspended)</span>}
            </h3>
            <div className="font-mono text-sm">
                Step: <span className="text-gray-300">{visData ? `'${visData.inputToken}' → '${visData.targetToken}'` : '...'}</span>
            </div>
        </div>
        <div className="flex justify-around items-center relative h-full">
            {/* Input Layer */}
            <div className="flex flex-col items-center space-y-2">
                <h4 className="text-sm font-semibold text-gray-400">Input</h4>
                <div className="p-2 bg-gray-900/50 rounded-lg border border-gray-700 min-h-[52px] flex items-center">
                    {inputToken ? (
                        <Neuron
                            label={inputToken}
                            activation={1}
                            isInput={true}
                            isTarget={false}
                            isPredicted={false}
                        />
                    ) : <div className="w-10 h-10"></div>}
                </div>
            </div>
            
            <div className="text-5xl text-gray-600 mx-2 sm:mx-4">→</div>

            {/* Hidden Layer */}
            <div className="flex flex-col items-center space-y-2">
                <h4 className="text-sm font-semibold text-gray-400">Hidden ({hiddenSize})</h4>
                <div className="flex flex-wrap justify-center items-center gap-1 p-2 bg-gray-900/50 rounded-lg border border-gray-700 max-w-[12rem] min-h-[52px]">
                    {hiddenLayerLabels.map((label, i) => {
                        const activation = activations?.hidden?.data?.[0]?.[i] ?? 0;
                        return (
                            <Neuron
                                key={i}
                                label={''} // Hide label to save space
                                activation={activation}
                                isInput={false}
                                isTarget={false}
                                isPredicted={false}
                            />
                        );
                    })}
                </div>
            </div>

            <div className="text-5xl text-gray-600 mx-2 sm:mx-4">→</div>

            {/* Output Layer */}
            <div className="flex flex-col items-center space-y-2">
                <h4 className="text-sm font-semibold text-gray-400">Output</h4>
                <div className="flex flex-col items-center p-2 space-y-2 bg-gray-900/50 rounded-lg border border-gray-700 min-h-[112px] justify-center">
                    {renderOutputNeurons()}
                </div>
            </div>

            {/* RNN/GRU Recurrent Connection */}
            {showRecurrent && (
                <div className="absolute top-1/2 left-1/2 -translate-x-[45%] -translate-y-[60%] sm:-translate-x-1/2 sm:-translate-y-1/2 pointer-events-none">
                    <svg width="100" height="100" viewBox="0 0 100 100" className="opacity-50">
                        <path d="M 50, 20 A 30,30 0 1,1 50, 80" stroke="#888" strokeWidth="2" fill="none" />
                        <path d="M 50, 80 L 45, 70 M 50, 80 L 55, 70" stroke="#888" strokeWidth="2" fill="none" />
                        <text x="60" y="55" fill="#888" fontSize="10" transform="rotate(90 60,55)">memory</text>
                    </svg>
                </div>
            )}
        </div>
        </div>
    </Tooltip>
  );
};