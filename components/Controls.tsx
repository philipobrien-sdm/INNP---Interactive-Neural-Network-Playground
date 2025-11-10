/**
 * @file Controls.tsx
 * @description A comprehensive component that renders all the user-configurable controls
 * for the training process. This includes starting/stopping training, resetting the model,
 * adjusting hyperparameters, and configuring cyclical training.
 */

import React from 'react';
import { PlayIcon, PauseIcon, ResetIcon, SaveIcon, UploadIcon } from './icons';
import { Tooltip } from './Tooltip';

/**
 * The Controls component provides the main user interface for interacting with the training process.
 * It's a "controlled component," meaning its state is managed by the parent Playground component
 * and passed down via props. This allows for a single source of truth for all training parameters.
 * @param {object} props - The component's props, including training state, event handlers, and parameter values.
 */
export const Controls = ({
  trainingState,
  onStart,
  onPause,
  onReset,
  onSaveModel,
  onLoadModel,
  model,
  trainingText,
  setTrainingText,
  learningRate,
  setLearningRate,
  hiddenSize,
  setHiddenSize,
  epochs,
  setEpochs,
  batchSize,
  setBatchSize,
  tokenizerType,
  setTokenizerType,
  customTokenizerSet,
  setCustomTokenizerSet,
  vocabSize,
  setVocabSize,
  dropoutRate,
  setDropoutRate,
  useAutoFastMode,
  setUseAutoFastMode,
  batchSizeLabel,
  batchSizeTooltip,
  modelType,
  isAutoCoaching,
  isCyclicalMode,
  setIsCyclicalMode,
  cycleEpochs,
  setCycleEpochs,
  numCycles,
  setNumCycles,
  currentCycle
}) => {
  // --- Derived State ---
  // These booleans simplify conditional rendering and logic based on the training state.
  const isTraining = trainingState === 'RUNNING';
  const isPaused = trainingState === 'PAUSED';
  const isFinished = trainingState === 'FINISHED';
  const isCoaching = trainingState === 'COACHING';
  // Disable most controls during any active process to prevent inconsistent states.
  const isDisabled = isTraining || isPaused || isAutoCoaching || isCoaching;

  /**
   * Determines the text for the main action button (Start/Pause/Resume).
   * @returns {string} The appropriate button label.
   */
  const getButtonText = () => {
    if (isTraining) return 'Pause';
    if (isPaused) return 'Resume';
    if (isFinished) return 'Continue Training';
    return 'Start Training';
  };
  
  /**
   * Determines the text for the status display.
   * @returns {string} The current training status.
   */
  const getStatusText = () => {
      if (isCoaching) return `COACHING ${currentCycle}/${numCycles}`;
      if (isCyclicalMode && isTraining) return `TRAINING ${currentCycle}/${numCycles}`;
      if (isAutoCoaching) return 'AUTO-COACHING';
      if (isTraining) return 'RUNNING';
      return trainingState;
  }

  return (
    <div className="space-y-6">
      {/* --- Main Action Buttons and Status Display --- */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <Tooltip text="Click to start, resume, or continue training.">
            <button
            onClick={isTraining ? onPause : onStart}
            className="w-full flex items-center justify-center px-4 py-2 rounded-md font-semibold transition-colors duration-200 text-white bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600"
            disabled={isAutoCoaching || isCoaching}
            >
            {isTraining ? <PauseIcon className="w-5 h-5 mr-2" /> : <PlayIcon className="w-5 h-5 mr-2" />}
            {getButtonText()}
            </button>
        </Tooltip>
        <Tooltip text="Stop training and reset the model and all parameters to their initial state.">
            <button
            onClick={onReset}
            disabled={isAutoCoaching || isCoaching}
            className="w-full flex items-center justify-center px-4 py-2 bg-red-600 hover:bg-red-700 rounded-md font-semibold transition-colors duration-200 text-white disabled:bg-gray-600"
            >
            <ResetIcon className="w-5 h-5 mr-2" />
            Reset
            </button>
        </Tooltip>
        <Tooltip text="Save the current trained model state and configuration to a JSON file.">
            <button
              onClick={onSaveModel}
              disabled={!model || isAutoCoaching || isCoaching}
              className="w-full flex items-center justify-center px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-md font-semibold transition-colors duration-200 text-white disabled:bg-gray-600 disabled:cursor-not-allowed"
            >
              <SaveIcon className="w-5 h-5 mr-2" />
              Save
            </button>
        </Tooltip>
        <Tooltip text="Load a previously saved model from a JSON file.">
            <button
              onClick={onLoadModel}
              disabled={isTraining || isPaused || isAutoCoaching || isCoaching}
              className="w-full flex items-center justify-center px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-md font-semibold transition-colors duration-200 text-white disabled:bg-gray-600 disabled:cursor-not-allowed"
            >
              <UploadIcon className="w-5 h-5 mr-2" />
              Load
            </button>
        </Tooltip>
        <div className="flex items-center justify-center font-mono text-center bg-gray-900 p-2 rounded-md border border-gray-700">
            <span className="text-gray-400 mr-2">Status:</span>
            <span className={`font-bold ${
                isCoaching ? 'text-orange-400' : isTraining ? 'text-green-400' : isPaused ? 'text-yellow-400' : isFinished ? 'text-blue-400' : 'text-gray-400'
            }`}>
            {getStatusText()}
            </span>
        </div>
      </div>

      {/* --- Training Data and Tokenizer Configuration --- */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-2">
            <Tooltip text="The text the model will learn from. You can edit this.">
                <label htmlFor="training-text" className="block text-sm font-medium text-gray-300">
                    Training Text
                </label>
            </Tooltip>
            <textarea
            id="training-text"
            value={trainingText}
            onChange={(e) => setTrainingText(e.target.value)}
            disabled={isDisabled}
            className="w-full h-40 p-2 bg-gray-900 border border-gray-700 rounded-md font-mono text-sm disabled:opacity-70"
            />
        </div>
        <div className="space-y-4">
            <div>
                <Tooltip text="Choose the tokenization method. This defines the basic units (the 'vocabulary') the model learns from.">
                    <label className="block text-sm font-medium text-gray-300 mb-2">Tokenizer</label>
                </Tooltip>
                <div className="flex space-x-4">
                    {['Character', 'BPE', 'Custom'].map(type => (
                        <div key={type} className="flex items-center">
                            <input
                                id={`tokenizer-${type}`}
                                name="tokenizer"
                                type="radio"
                                checked={tokenizerType === type.toLowerCase()}
                                onChange={() => setTokenizerType(type.toLowerCase())}
                                disabled={isDisabled}
                                className="h-4 w-4 text-cyan-600 bg-gray-800 border-gray-600 focus:ring-cyan-500"
                            />
                            <label htmlFor={`tokenizer-${type}`} className="ml-2 block text-sm text-gray-300">{type}</label>
                        </div>
                    ))}
                </div>
            </div>
            {tokenizerType === 'bpe' && (
                <div className="space-y-2 pl-2">
                    <Tooltip text="Controls how many sub-word merges are learned by the BPE algorithm.">
                        <label htmlFor="vocab-size" className="block text-sm font-medium text-gray-300">
                            BPE Vocab Size: <span className="font-mono bg-gray-900 px-2 py-1 rounded-md">{vocabSize}</span>
                        </label>
                    </Tooltip>
                    <input
                        id="vocab-size"
                        type="range" min="257" max="1024" step="1"
                        value={vocabSize}
                        onChange={(e) => setVocabSize(parseInt(e.target.value))}
                        disabled={isDisabled}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
                    />
                </div>
            )}
            {tokenizerType === 'custom' && (
                <div className="space-y-2 pl-2">
                    <Tooltip text="A comma-separated list of custom tokens. The tokenizer will match the longest tokens first. Any single characters from the training text not covered by these tokens will be added to the vocabulary automatically.">
                        <label htmlFor="custom-set" className="block text-sm font-medium text-gray-300">
                            Custom Token Set
                        </label>
                    </Tooltip>
                    <textarea
                        id="custom-set"
                        value={customTokenizerSet}
                        onChange={(e) => setCustomTokenizerSet(e.target.value)}
                        disabled={isDisabled}
                        className="w-full h-20 p-2 bg-gray-900 border border-gray-700 rounded-md font-mono text-xs disabled:opacity-70"
                    />
                </div>
            )}
            <Tooltip text="Automatically speed up training after 60s by suspending UI updates. Training remains fully visualized for the final epoch.">
              <div className="flex items-center space-x-2">
                  <input 
                      type="checkbox" 
                      id="use-fast-mode" 
                      checked={useAutoFastMode} 
                      onChange={(e) => setUseAutoFastMode(e.target.checked)}
                      disabled={isDisabled}
                      className="h-4 w-4 rounded border-gray-600 bg-gray-800 text-cyan-600 focus:ring-cyan-500"
                  />
                  <label htmlFor="use-fast-mode" className="text-sm font-medium text-gray-300">
                      Enable Automatic Fast Mode
                  </label>
              </div>
          </Tooltip>
        </div>
      </div>
      
      {/* --- Hyperparameter Sliders --- */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="space-y-2">
              <Tooltip text="Controls how much the model adjusts its weights on each update. Too high can be unstable, too low can be very slow.">
                <label htmlFor="learning-rate" className="block text-sm font-medium text-gray-300">
                    Learning Rate: <span className="font-mono bg-gray-900 px-2 py-1 rounded-md">{learningRate.toExponential(1)}</span>
                </label>
              </Tooltip>
              <input
                  id="learning-rate"
                  type="range" min="0.0001" max="0.1" step="0.0001"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  disabled={isDisabled}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
              />
          </div>
          <div className="space-y-2">
              <Tooltip text="The number of neurons in the 'thinking' layer of the network. More neurons can learn more complex patterns but take longer to train.">
                <label htmlFor="hidden-size" className="block text-sm font-medium text-gray-300">
                    Hidden Size: <span className="font-mono bg-gray-900 px-2 py-1 rounded-md">{hiddenSize}</span>
                </label>
              </Tooltip>
              <input
                  id="hidden-size"
                  type="range" min="4" max="128" step="4"
                  value={hiddenSize}
                  onChange={(e) => setHiddenSize(parseInt(e.target.value))}
                  disabled={isDisabled}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
              />
          </div>
          <div className="space-y-2">
             <Tooltip text="The maximum number of times the model will process the entire training text. Training may stop early if the model's performance plateaus.">
                <label htmlFor="epochs" className="block text-sm font-medium text-gray-300">
                    Max Epochs: <span className="font-mono bg-gray-900 px-2 py-1 rounded-md">{epochs}</span>
                </label>
              </Tooltip>
              <input
                  id="epochs"
                  type="range" min="1" max="1000" step="1"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  disabled={isDisabled || isCyclicalMode} // Disabled during cyclical mode as it's controlled by cycles.
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
              />
          </div>
          <div className="space-y-2">
              <Tooltip text={batchSizeTooltip}>
                <label htmlFor="batch-size" className="block text-sm font-medium text-gray-300">
                    {batchSizeLabel}: <span className="font-mono bg-gray-900 px-2 py-1 rounded-md">{batchSize}</span>
                </label>
              </Tooltip>
              <input
                  id="batch-size"
                  type="range" min="4" max={modelType === 'RNN' || modelType === 'GRU' ? 64 : 128} step="4"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  disabled={isDisabled}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
              />
          </div>
      </div>
       {/* Dropout control, only shown for recurrent models (RNN, GRU, LSTM) */}
       {(modelType === 'RNN' || modelType === 'GRU' || modelType === 'LSTM') && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="space-y-2 lg:col-start-4">
                  <Tooltip text="The probability of dropping a neuron's output during training. Helps prevent overfitting.">
                    <label htmlFor="dropout-rate" className="block text-sm font-medium text-gray-300">
                        Dropout Rate: <span className="font-mono bg-gray-900 px-2 py-1 rounded-md">{dropoutRate.toFixed(2)}</span>
                    </label>
                  </Tooltip>
                  <input
                      id="dropout-rate"
                      type="range" min="0" max="0.5" step="0.05"
                      value={dropoutRate}
                      onChange={(e) => setDropoutRate(parseFloat(e.target.value))}
                      disabled={isDisabled}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
                  />
              </div>
          </div>
      )}
      {/* --- Cyclical Training Configuration --- */}
      <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 space-y-4">
          <Tooltip text="Automate a cycle of training, followed by auto-coaching, and repeat.">
              <div className="flex items-center space-x-2">
                  <input 
                      type="checkbox" 
                      id="use-cyclical-mode" 
                      checked={isCyclicalMode} 
                      onChange={(e) => setIsCyclicalMode(e.target.checked)}
                      disabled={isDisabled}
                      className="h-4 w-4 rounded border-gray-600 bg-gray-800 text-cyan-600 focus:ring-cyan-500"
                  />
                  <label htmlFor="use-cyclical-mode" className="text-sm font-medium text-gray-300">
                      Enable Cyclical Training (Train ↔ Coach)
                  </label>
              </div>
          </Tooltip>
          {isCyclicalMode && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pl-6">
                   <div className="space-y-2">
                      <Tooltip text="The number of training epochs to run before each 60-second auto-coaching session.">
                        <label htmlFor="cycle-epochs" className="block text-sm font-medium text-gray-300">
                            Epochs per Cycle: <span className="font-mono bg-gray-900 px-2 py-1 rounded-md">{cycleEpochs}</span>
                        </label>
                      </Tooltip>
                      <input
                          id="cycle-epochs"
                          type="range" min="1" max="50" step="1"
                          value={cycleEpochs}
                          onChange={(e) => setCycleEpochs(parseInt(e.target.value))}
                          disabled={isDisabled}
                          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
                      />
                  </div>
                   <div className="space-y-2">
                      <Tooltip text="The total number of train/coach cycles to run.">
                        <label htmlFor="num-cycles" className="block text-sm font-medium text-gray-300">
                            Number of Cycles: <span className="font-mono bg-gray-900 px-2 py-1 rounded-md">{numCycles}</span>
                        </label>
                      </Tooltip>
                      <input
                          id="num-cycles"
                          type="range" min="1" max="20" step="1"
                          value={numCycles}
                          onChange={(e) => setNumCycles(parseInt(e.target.value))}
                          disabled={isDisabled}
                          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
                      />
                  </div>
              </div>
          )}
      </div>
    </div>
  );
};
