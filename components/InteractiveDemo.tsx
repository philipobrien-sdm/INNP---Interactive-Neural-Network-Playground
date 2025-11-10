/**
 * @file InteractiveDemo.tsx
 * @description A component that provides a highly detailed, step-by-step, interactive
 * visualization of a single FFNN training on a very small, predictable dataset ("abcabc").
 * This is designed for educational purposes to show the exact matrix math involved in
 * both the forward and backward passes.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { initializeFFNNModel, trainStepFFNN, generateFFNN } from '../services/languageModel';
import { FFNNModel, TrainStepResult, Matrix } from '../types';
import { LineArchitectureVisualizer } from './LineArchitectureVisualizer';
import { PlayIcon, PauseIcon, ResetIcon, SparklesIcon } from './icons';
import { Tooltip } from './Tooltip';
import { LossHistogram } from './LossHistogram';

// --- Constants for the Demo ---
const INTERACTIVE_TEXT = "abcabc";
const INTERACTIVE_LR = 0.1;
const INTERACTIVE_HIDDEN_SIZE = 4;
const MAX_EPOCHS = 50;
const AUTOPLAY_DELAY = 150; // ms between steps

// Defines the possible states of the interactive demo.
type DemoState = 'IDLE' | 'RUNNING' | 'PAUSED' | 'FINISHED';

// Interface for storing generated text at different epochs.
interface GenerationHistory {
    epoch: number;
    text: string;
}

// --- Start of MatrixDisplay component ---
// A reusable component to display a matrix with optional highlighting.

interface Highlight {
    index: number;
    color: string;
    label?: string;
}

interface MatrixDisplayProps {
    matrix: Matrix;
    title: string;
    description?: string;
    highlights?: Highlight[];
}

const MatrixDisplay: React.FC<MatrixDisplayProps> = ({ matrix, title, description, highlights = [] }) => {
    if (!matrix) return null;

    const { rows, cols, data } = matrix;

    const getFontSize = () => {
        if (cols > 10) return 'text-[10px]';
        return 'text-xs';
    };
    
    // Use a Map for efficient lookup of highlights by cell index.
    const highlightMap = new Map<number, { color: string, label?: string }>();
    highlights.forEach(h => highlightMap.set(h.index, { color: h.color, label: h.label }));

    return (
        <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
            <h4 className="text-md font-semibold text-gray-300 mb-2">{title} <span className="font-mono text-gray-500 text-sm">({rows}x{cols})</span></h4>
            {description && <p className="text-sm text-gray-400 mb-3">{description}</p>}
            <div 
                className="grid gap-px bg-gray-700 p-px rounded-md overflow-x-auto"
                style={{ gridTemplateColumns: `repeat(${cols}, minmax(45px, 1fr))` }}
            >
                {data.flat().map((val, index) => {
                    const highlight = highlightMap.get(index);
                    const bgColor = highlight ? highlight.color : 'bg-gray-800';
                    return (
                        <div 
                            key={index} 
                            className={`relative p-1 text-center font-mono ${getFontSize()} ${bgColor}`}
                            title={val.toString()}
                        >
                            {highlight?.label && (
                                <span className="absolute top-0 left-0 text-[8px] font-bold text-white px-1 bg-black/50 rounded-br-sm">{highlight.label}</span>
                            )}
                            {val.toFixed(3)}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};
// --- End of MatrixDisplay component ---


export const InteractiveDemo = () => {
    // --- State Management ---
    const [model, setModel] = useState<FFNNModel | null>(null);
    const [encodedText, setEncodedText] = useState<number[]>([]);
    const [step, setStep] = useState(0); // Current position in the training text.
    const [epoch, setEpoch] = useState(1); // Current epoch.
    const [visData, setVisData] = useState<TrainStepResult | null>(null); // Data from the last training step for visualization.
    const [log, setLog] = useState<string[]>([]);
    const [trainingState, setTrainingState] = useState<DemoState>('IDLE');
    const [lossHistory, setLossHistory] = useState<number[]>([]);
    const [generationHistory, setGenerationHistory] = useState<GenerationHistory[]>([]);
    
    // Refs to store loss data for the current epoch without causing re-renders.
    const currentEpochLoss = useRef(0);
    const currentEpochSteps = useRef(0);

    /**
     * Initializes or resets the entire demo to its starting state.
     */
    const initialize = useCallback(() => {
        const vocab = [...new Set(INTERACTIVE_TEXT.split(''))].sort();
        const newModel = initializeFFNNModel(vocab, INTERACTIVE_HIDDEN_SIZE);
        setModel(newModel);

        const encoded = INTERACTIVE_TEXT.split('').map(char => newModel.tokenToIndex[char]);
        setEncodedText(encoded);
        setStep(0);
        setEpoch(1);
        setVisData(null);
        setLossHistory([]);
        setGenerationHistory([]);
        setTrainingState('IDLE');
        currentEpochLoss.current = 0;
        currentEpochSteps.current = 0;
        setLog(['Initialized model. Press "Next Step" or "Autoplay" to begin training.']);
    }, []);

    // Initialize the demo when the component mounts.
    useEffect(() => {
        initialize();
    }, [initialize]);

    /**
     * Executes a single training step.
     */
    const handleRunStep = useCallback(() => {
        if (trainingState === 'FINISHED' || !model) return;
        
        let currentStep = step;
        let currentEpoch = epoch;

        // If we've reached the end of the text, an epoch is complete.
        if (currentStep >= encodedText.length - 1) {
            // Calculate and record the average loss for the completed epoch.
            const avgLoss = currentEpochSteps.current > 0 ? currentEpochLoss.current / currentEpochSteps.current : 0;
            setLossHistory(prev => [...prev, avgLoss]);

            // Generate a test output every few epochs to see progress.
            if (epoch % 5 === 0 && epoch > 0) {
                if(model) {
                    const seed = INTERACTIVE_TEXT[0] || 'a';
                    const generated = generateFFNN(model, seed, 20, 0.7);
                    setGenerationHistory(prev => [...prev, { epoch: epoch, text: generated }]);
                }
            }

            // Reset for the next epoch.
            currentEpoch += 1;
            currentStep = 0;
            currentEpochLoss.current = 0;
            currentEpochSteps.current = 0;

            if (currentEpoch > MAX_EPOCHS) {
                setTrainingState('FINISHED');
                setLog(prev => [`Training finished after ${MAX_EPOCHS} epochs.`, ...prev].slice(0, 100));
                return;
            }
        }
        
        // Perform the actual training step.
        const result = trainStepFFNN(model, encodedText, currentStep, 1, INTERACTIVE_LR);
        
        // Update state with the results.
        setModel(result.updatedModel as FFNNModel);
        setVisData(result);
        setStep(currentStep + 1);
        setEpoch(currentEpoch);

        // Accumulate loss for the current epoch.
        currentEpochLoss.current += result.loss;
        currentEpochSteps.current += 1;

        // Update the log.
        const { inputToken, targetToken, predictedToken, loss } = result;
        setLog(prev => [
            `[E:${currentEpoch}, S:${currentStep + 1}] Input: '${inputToken}', Target: '${targetToken}', Pred: '${predictedToken}', Loss: ${loss.toFixed(3)}`,
            ...prev
        ].slice(0, 100));
    }, [model, encodedText, step, epoch, trainingState]);
    
    /**
     * Effect to handle the "Autoplay" functionality.
     */
    useEffect(() => {
        if (trainingState !== 'RUNNING') {
            return; // Do nothing if not running.
        }
        // When running, automatically call the next step after a delay.
        const timeoutId = setTimeout(handleRunStep, AUTOPLAY_DELAY);
        // Cleanup function to cancel the timeout if the component unmounts or state changes.
        return () => clearTimeout(timeoutId);
    }, [trainingState, step, handleRunStep]);


    const handlePlay = () => {
        setTrainingState('RUNNING');
    };
    
    const handlePause = () => {
        setTrainingState('PAUSED');
    };

    /**
     * Manually triggers a generation test.
     */
    const handleTestOutput = () => {
        if (!model) return;
        const seed = INTERACTIVE_TEXT[0] || 'a';
        const generated = generateFFNN(model, seed, 20, 0.7);
        setLog(prev => [`[TEST OUTPUT]: ${generated}`, ...prev].slice(0, 100));
    };
    
    // --- Derived state and data for rendering ---
    const isRunning = trainingState === 'RUNNING';
    const isFinished = trainingState === 'FINISHED';

    const { inputToken, targetToken, predictedToken, activations, gradients, loss } = visData || {};
    const inputIndex = (model && inputToken) ? model.tokenToIndex[inputToken] : -1;
    const targetIndex = (model && targetToken) ? model.tokenToIndex[targetToken] : -1;
    const predictedIndex = (model && predictedToken) ? model.tokenToIndex[predictedToken] : -1;

    // Construct the one-hot input vector for display.
    let inputVector: Matrix | null = null;
    if (model && inputIndex !== -1) {
        const data = Array(model.vocab.length).fill(0);
        data[inputIndex] = 1;
        inputVector = { rows: 1, cols: model.vocab.length, data: [data] };
    }
    
    // Prepare highlights for the matrix displays.
    const inputHighlights: Highlight[] = [];
    if (inputIndex !== -1) {
        inputHighlights.push({ index: inputIndex, color: 'bg-cyan-700/50', label: 'Input' });
    }

    const outputHighlights: Highlight[] = [];
    if (predictedIndex === targetIndex && targetIndex !== -1) {
        outputHighlights.push({ index: targetIndex, color: 'bg-green-700/60', label: 'Correct ✓' });
    } else {
        if (targetIndex !== -1) {
            outputHighlights.push({ index: targetIndex, color: 'bg-green-700/60', label: 'Target' });
        }
        if (predictedIndex !== -1) {
            outputHighlights.push({ index: predictedIndex, color: 'bg-yellow-700/60', label: 'Predicted' });
        }
    }

    return (
        <div className="space-y-6">
            <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                <h2 className="text-xl font-semibold mb-2 text-cyan-400">Interactive Demo (FFNN)</h2>
                <p className="text-gray-400">
                    Welcome to the Interactive Demo! This is the best place to start. It provides a slow, clear, and detailed visualization of a single training step for our simplest model, the Feed-Forward Neural Network (FFNN). Unlike the other tabs, this demo focuses on a tiny, predictable dataset ("{INTERACTIVE_TEXT}") to show you the exact matrix math involved in making a prediction (the forward pass) and learning from a mistake (the backward pass). The key takeaway: an FFNN has no memory; it only sees one character at a time.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-2 space-y-6">
                    {/* Visualizes the network architecture with connections. */}
                    <LineArchitectureVisualizer model={model} visData={visData} />
                    
                    {/* This section shows the detailed mathematical breakdown of a single step. */}
                    {visData && model && (
                        <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 space-y-6">
                            <h3 className="text-xl font-semibold text-cyan-400 border-b border-gray-700 pb-2">Step-by-Step Calculation Breakdown</h3>
                            
                            {/* --- Forward Pass Visualization --- */}
                            <div>
                                <h4 className="text-lg font-semibold text-gray-200 mb-2">1. Forward Pass: Making a Prediction</h4>
                                {inputVector && (
                                    <MatrixDisplay
                                        matrix={inputVector}
                                        title="Input Vector"
                                        description={`The character '${inputToken}' is converted into a one-hot vector. This is the input to the network.`}
                                        highlights={inputHighlights}
                                    />
                                )}
                                <div className="text-center text-2xl my-4 text-gray-500">↓</div>
                                {activations?.hidden && (
                                    <MatrixDisplay
                                        matrix={activations.hidden}
                                        title="Hidden Layer Activation"
                                        description={`Calculated as: tanh(input_vector · W_hidden + b_hidden). This is the network's internal representation or "thought" about the input.`}
                                    />
                                )}
                                <div className="text-center text-2xl my-4 text-gray-500">↓</div>
                                {activations?.outputRaw && (
                                    <>
                                        <MatrixDisplay
                                            matrix={activations.outputRaw}
                                            title="Output Logits (Raw Scores)"
                                            description={`Calculated as: hidden_activation · W_output + b_output. These raw scores can be positive or negative. The highest *value* (not magnitude) determines the most likely next character.`}
                                        />
                                        <div className="text-center text-2xl my-4 text-gray-500">↓</div>
                                    </>
                                )}
                                {activations?.output && (
                                    <MatrixDisplay
                                        matrix={activations.output}
                                        title="Output Probabilities"
                                        description={`Calculated as: softmax(logits). The softmax function exponentiates the logits (making them all positive) and normalizes them into probabilities that sum to 1. The highest probability is the model's final prediction ('${predictedToken}'). The target was '${targetToken}'.`}
                                        highlights={outputHighlights}
                                    />
                                )}
                            </div>
                            
                            {/* --- Loss Visualization --- */}
                            <div className="text-center font-mono text-lg p-2 bg-gray-900 rounded-md">
                                Loss: <span className="text-yellow-400 font-bold">{loss?.toFixed(4)}</span>
                                <p className="text-sm text-gray-400 mt-1">Calculated as -log(probability of target). A lower loss means a more confident, correct prediction.</p>
                            </div>

                            {/* --- Backward Pass Visualization --- */}
                            {gradients && (
                                 <div>
                                    <h4 className="text-lg font-semibold text-gray-200 mb-2">2. Backward Pass & Weight Update</h4>
                                    <p className="text-sm text-gray-400 mb-4">
                                        The model calculates gradients, which measure how much each weight and bias contributed to the error (loss). These gradients are then used to update the weights to improve future predictions via the formula: `new_weight = old_weight - learning_rate * gradient`.
                                    </p>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        <MatrixDisplay
                                            matrix={gradients.outputLayer.weights}
                                            title="Output Weight Gradients (∇W_o)"
                                        />
                                         <MatrixDisplay
                                            matrix={gradients.outputLayer.biases}
                                            title="Output Bias Gradients (∇b_o)"
                                        />
                                         <MatrixDisplay
                                            matrix={gradients.hiddenLayer.weights}
                                            title="Hidden Weight Gradients (∇W_h)"
                                        />
                                         <MatrixDisplay
                                            matrix={gradients.hiddenLayer.biases}
                                            title="Hidden Bias Gradients (∇b_h)"
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
                {/* --- Controls and Log Panel --- */}
                <div className="space-y-4 flex flex-col">
                     <div className="grid grid-cols-2 gap-2">
                        <Tooltip text="Reset the model to its initial random state.">
                            <button onClick={initialize} className="flex-1 flex items-center justify-center px-4 py-2 bg-red-600 hover:bg-red-700 rounded-md font-semibold text-white">
                                <ResetIcon className="w-5 h-5 mr-2" />
                                Reset
                            </button>
                        </Tooltip>
                        <Tooltip text={isRunning ? "Pause automatic training" : "Automatically run through training steps"}>
                             <button onClick={isRunning ? handlePause : handlePlay} disabled={isFinished} className="flex-1 flex items-center justify-center px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-md font-semibold text-white disabled:bg-gray-600">
                                {isRunning ? <PauseIcon className="w-5 h-5 mr-2" /> : <PlayIcon className="w-5 h-5 mr-2" />}
                                {isRunning ? 'Pause' : 'Autoplay'}
                            </button>
                        </Tooltip>
                        <Tooltip text="Process the next character in the sequence.">
                             <button onClick={handleRunStep} disabled={isFinished || isRunning} className="flex-1 flex items-center justify-center px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-md font-semibold text-white disabled:bg-gray-600">
                                Next Step
                            </button>
                        </Tooltip>
                        <Tooltip text="Generate a sample output from the model in its current state.">
                            <button onClick={handleTestOutput} disabled={!model} className="flex-1 flex items-center justify-center px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-md font-semibold text-white disabled:bg-gray-600">
                                <SparklesIcon className="w-5 h-5 mr-2" />
                                Test Output
                            </button>
                        </Tooltip>
                    </div>
                     <div className="flex-grow h-80 overflow-y-auto bg-gray-900 p-3 rounded-md border border-gray-700">
                      <h3 className="text-lg font-semibold text-cyan-400 mb-2">Log</h3>
                      <ul className="text-xs font-mono text-gray-400 space-y-1">
                        {log.map((entry, i) => (
                          <li key={i} className={`whitespace-pre-wrap ${i === 0 ? 'text-gray-100' : ''}`}>
                            {entry}
                          </li>
                        ))}
                      </ul>
                    </div>
                </div>
            </div>
            {/* --- Charts --- */}
            <div className="space-y-6">
                <LossHistogram lossHistory={lossHistory} />
                {generationHistory.length > 0 && (
                    <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                        <h2 className="text-xl font-semibold mb-4 text-cyan-400">Test Output History</h2>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm text-left text-gray-400">
                                <thead className="text-xs text-gray-300 uppercase bg-gray-700">
                                    <tr>
                                        <th scope="col" className="px-6 py-3 w-1/4">
                                            Epoch
                                        </th>
                                        <th scope="col" className="px-6 py-3">
                                            Generated Text
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {generationHistory.map((item, index) => (
                                        <tr key={index} className="bg-gray-800 border-b border-gray-700 hover:bg-gray-700/50">
                                            <td className="px-6 py-4 font-medium text-gray-200">
                                                {item.epoch}
                                            </td>
                                            <td className="px-6 py-4 font-mono text-cyan-300">
                                                {item.text}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};