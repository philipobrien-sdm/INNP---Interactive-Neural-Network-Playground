/**
 * @file Playground.tsx
 * @description This is the primary stateful component that orchestrates the entire training
 * and visualization process for a given model type. It manages all state related to the
 * model, training progress, hyperparameters, logs, and visualizations.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Controls } from './Controls';
import { LogPanel } from './LogPanel';
import { GenerationPanel } from './GenerationPanel';
import { ArchitectureVisualizer } from './ArchitectureVisualizer';
import { LossHistogram } from './LossHistogram';
import { GenerationHistoryPanel } from './GenerationHistoryPanel';
import { SuccessRateHeatmap } from './SuccessRateHeatmap';
import {
  initializeFFNNModel, trainStepFFNN, generateFFNN,
  initializeRNNModel, trainStepRNN, generateRNN,
  initializeGRUModel, trainStepGRU, generateGRU,
  initializeLSTMModel, trainStepLSTM, generateLSTM
} from '../services/languageModel';
import { trainBPE, encodeBPE } from '../services/bpe';
import { encodeCustom } from '../services/customTokenizer';
import { isGoodWord } from '../services/wordValidator';
import { ALL_TOKENS_STRING, ONSETS, VOWELS } from '../services/phonotactics';
import { LanguageModel, TrainStepResult, GenerationHistoryItem, BpeMerges } from '../types';
import {
  DEFAULT_TRAINING_TEXT,
  EARLY_STOPPING_PATIENCE,
  FINAL_LR_MULTIPLIER
} from '../constants';

// Defines the possible states of the training process.
type TrainingState = 'IDLE' | 'RUNNING' | 'PAUSED' | 'FINISHED' | 'STABILIZED' | 'FINE_TUNING' | 'COACHING';
// Defines the available tokenizer types.
type TokenizerType = 'character' | 'bpe' | 'custom';

// Props for the Playground component, allowing it to be configured for different model architectures.
interface PlaygroundProps {
  modelType: 'FFNN' | 'RNN' | 'GRU' | 'LSTM';
  batchSizeLabel: string;
  batchSizeTooltip: string;
  defaultLearningRate: number;
  defaultHiddenSize: number;
  defaultEpochs: number;
  defaultBatchSize: number;
}

// Type definitions for tracking prediction statistics.
type PredictionStats = {
  [fromToken: string]: {
    [toToken: string]: {
      correct: number;
      total: number;
    };
  };
};

// Represents a word generated during auto-coaching and whether it was deemed "good".
interface AutoCoachedWord {
    word: string;
    isGood: boolean;
}

// Defines the possible open tabs in the right-hand accordion UI.
type AccordionTab = 'log' | 'generate' | 'history' | null;

// FIX: Explicitly typed AccordionItem props to resolve issue with 'children' prop type inference.
interface AccordionItemProps {
  title: string;
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

/**
 * A reusable accordion item component for the sidebar UI.
 */
const AccordionItem: React.FC<AccordionItemProps> = ({ title, isOpen, onToggle, children }) => {
    return (
        <div className="bg-gray-800 rounded-lg border border-gray-700">
            <button
                onClick={onToggle}
                className="w-full flex justify-between items-center p-4 text-left focus:outline-none focus:ring-2 focus:ring-cyan-500 rounded-t-lg"
                aria-expanded={isOpen}
            >
                <h2 className="text-xl font-semibold text-cyan-400">{title}</h2>
                <span className={`text-gray-400 transform transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}>▼</span>
            </button>
            {isOpen && (
                <div className="p-4 border-t border-gray-700">
                    {children}
                </div>
            )}
        </div>
    );
};


export const Playground: React.FC<PlaygroundProps> = ({ 
  modelType, 
  batchSizeLabel, 
  batchSizeTooltip,
  defaultLearningRate,
  defaultHiddenSize,
  defaultEpochs,
  defaultBatchSize
}) => {
  // --- STATE MANAGEMENT ---
  // The React `useState` hook is used to manage all the component's state.
  // When state is updated with a `set...` function, React re-renders the component.

  // Core training state
  const [trainingState, setTrainingState] = useState<TrainingState>('IDLE');
  const [model, setModel] = useState<LanguageModel | null>(null);
  
  // Hyperparameters
  const [trainingText, setTrainingText] = useState(DEFAULT_TRAINING_TEXT);
  const [learningRate, setLearningRate] = useState(defaultLearningRate);
  const [hiddenSize, setHiddenSize] = useState(defaultHiddenSize);
  const [epochs, setEpochs] = useState(defaultEpochs);
  const [batchSize, setBatchSize] = useState(defaultBatchSize);
  const [dropoutRate, setDropoutRate] = useState(0.1);
  
  // Tokenizer settings
  const [tokenizerType, setTokenizerType] = useState<TokenizerType>('custom');
  const [customTokenizerSet, setCustomTokenizerSet] = useState(ALL_TOKENS_STRING);
  const [vocabSize, setVocabSize] = useState(512);

  // UI and feature flags
  const [useAutoFastMode, setUseAutoFastMode] = useState(true);
  const [isFastMode, setIsFastMode] = useState(false);
  const [coachingEnabled, setCoachingEnabled] = useState(false);
  const [isAutoCoaching, setIsAutoCoaching] = useState(false);
  const [autoCoachedWords, setAutoCoachedWords] = useState<AutoCoachedWord[]>([]);
  const [modelIsLoaded, setModelIsLoaded] = useState(false);

  // Data for visualizations and logs
  const [logs, setLogs] = useState<string[]>([]);
  const [visData, setVisData] = useState<TrainStepResult | null>(null);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [generationHistory, setGenerationHistory] = useState<GenerationHistoryItem[]>([]);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [currentLearningRate, setCurrentLearningRate] = useState(learningRate);
  const [historicalPredictionStats, setHistoricalPredictionStats] = useState<PredictionStats[]>([]);
  
  // Cyclical Training State
  const [isCyclicalMode, setIsCyclicalMode] = useState(false);
  const [cycleEpochs, setCycleEpochs] = useState(10);
  const [numCycles, setNumCycles] = useState(5);
  const [currentCycle, setCurrentCycle] = useState(0);
  const [newWordsLog, setNewWordsLog] = useState<string[]>([]);
  const [coachingStatsLog, setCoachingStatsLog] = useState<string[]>([]);
  
  // Accordion UI State
  const [openAccordion, setOpenAccordion] = useState<AccordionTab>('log');


  // --- REFS FOR TRAINING LOOP ---
  // `useRef` is used to hold values that can change without triggering a re-render.
  // This is crucial for the high-frequency training loop, as re-rendering on every step
  // would be extremely slow. Refs provide a way to access the latest state values
  // from within the `requestAnimationFrame` callback.
  const trainingStateRef = useRef(trainingState);
  useEffect(() => { trainingStateRef.current = trainingState; }, [trainingState]);

  const isAutoCoachingRef = useRef(isAutoCoaching);
  useEffect(() => { isAutoCoachingRef.current = isAutoCoaching; }, [isAutoCoaching]);

  const loopIdRef = useRef<number | null>(null);
  const modelRef = useRef(model);
  useEffect(() => { modelRef.current = model; }, [model]);
  
  const encodedTextRef = useRef<number[]>([]);
  const currentStepRef = useRef(0);
  const currentEpochLossRef = useRef<number[]>([]);
  const bestLossRef = useRef(Infinity);
  const epochsWithoutImprovementRef = useRef(0);
  const fineTuningEpochsRemainingRef = useRef(0);
  const fastModeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const predictionStatsRef = useRef<PredictionStats>({});

  const bpeMergesRef = useRef<BpeMerges | null>(null);
  const bpeVocabRef = useRef<{ [key: number]: string } | null>(null);

  // Refs for cyclical training
  const isCyclicalModeRef = useRef(isCyclicalMode);
  useEffect(() => { isCyclicalModeRef.current = isCyclicalMode; }, [isCyclicalMode]);
  const cycleEpochsRef = useRef(cycleEpochs);
  useEffect(() => { cycleEpochsRef.current = cycleEpochs; }, [cycleEpochs]);
  const numCyclesRef = useRef(numCycles);
  useEffect(() => { numCyclesRef.current = numCycles; }, [numCycles]);
  const currentCycleRef = useRef(currentCycle);
  useEffect(() => { currentCycleRef.current = currentCycle; }, [currentCycle]);
  const coachingSessionStats = useRef({ good: 0, bad: 0 });
  const originalTrainingTextRef = useRef('');

  // --- HANDLERS AND LOGIC ---
  /**
   * Reinforces the model by performing several targeted training steps on a given text.
   * This is used for both manual and automated coaching.
   * @param {string} text - The word/text to reinforce.
   * @param {boolean} doLog - Whether to add messages to the main training log.
   */
  const handleReinforcement = useCallback((text: string, doLog = true) => {
    const currentModel = modelRef.current;
    if (!currentModel) return;

    if (doLog) {
        setLogs(prev => [`Reinforcing with: "${text}"...`, ...prev].slice(0, 50));
    }

    // Encode the reinforcement text using the current tokenizer.
    let encodedReinforcementText: number[];
    switch(tokenizerType) {
        case 'bpe':
            encodedReinforcementText = bpeMergesRef.current ? encodeBPE(text, bpeMergesRef.current) : [];
            break;
        case 'custom':
            encodedReinforcementText = encodeCustom(text, currentModel.vocab, currentModel.tokenToIndex);
            break;
        case 'character':
        default:
            encodedReinforcementText = text.split('').map(char => currentModel.tokenToIndex[char]).filter(id => id !== undefined);
            break;
    }

    if(encodedReinforcementText.length <= 1) return;

    // Run a mini-training loop on the reinforcement text.
    let reinforcedModel = currentModel;
    for(let i = 0; i < 5; i++) { // Reinforce for 5 iterations
      for(let j = 0; j < encodedReinforcementText.length - 1; j++) {
          let result;
          // Call the correct train step function for the current model type.
          switch(modelType) {
              case 'RNN':
                  result = trainStepRNN(reinforcedModel as any, encodedReinforcementText, j, 1, currentLearningRate * 0.5, dropoutRate);
                  break;
              case 'GRU':
                  result = trainStepGRU(reinforcedModel as any, encodedReinforcementText, j, 1, currentLearningRate * 0.5, dropoutRate);
                  break;
              case 'LSTM':
                  result = trainStepLSTM(reinforcedModel as any, encodedReinforcementText, j, 1, currentLearningRate * 0.5, dropoutRate);
                  break;
              case 'FFNN':
              default:
                  result = trainStepFFNN(reinforcedModel as any, encodedReinforcementText, j, 1, currentLearningRate * 0.5);
                  break;
          }
          reinforcedModel = result.updatedModel;
          // Update visualization data to show the reinforcement step.
          if (doLog || isAutoCoachingRef.current) {
            setVisData(result);
          }
      }
    }
    // Update the main model reference with the reinforced model.
    modelRef.current = reinforcedModel;
    setModel(reinforcedModel);
    
    if (doLog) {
        setLogs(prev => [`Reinforcement complete.`, ...prev].slice(0, 50));
    }
  }, [modelType, tokenizerType, currentLearningRate, dropoutRate]);

  /**
   * Pauses the training loop.
   */
  const handlePause = useCallback(() => {
    setTrainingState('PAUSED');
    if (fastModeTimeoutRef.current) clearTimeout(fastModeTimeoutRef.current);
  }, []);

  /**
   * Initializes or resets the entire training environment.
   * This function sets up the model, tokenizer, and all state variables.
   * @param {string} textToTrain - The text to use for training.
   * @param {boolean} keepParams - If true, keeps the current hyperparameter settings.
   */
  const initialize = useCallback(async (textToTrain = trainingText, keepParams = false) => {
    setModelIsLoaded(false);
    setTrainingState('IDLE');
    setLogs(['Model initialized. Ready to train.']);
    setVisData(null);
    setLossHistory([]);
    setGenerationHistory([]);
    setCurrentEpoch(1);
    currentStepRef.current = 0;
    currentEpochLossRef.current = [];
    bestLossRef.current = Infinity;
    epochsWithoutImprovementRef.current = 0;
    fineTuningEpochsRemainingRef.current = 0;
    setIsFastMode(false);
    setCoachingEnabled(false);
    setIsAutoCoaching(false);
    setAutoCoachedWords([]);
    predictionStatsRef.current = {};
    setHistoricalPredictionStats([]);
    
    // Reset cyclical state
    setCurrentCycle(0);
    setNewWordsLog([]);
    setCoachingStatsLog([]);

    if (fastModeTimeoutRef.current) clearTimeout(fastModeTimeoutRef.current);

    // --- Tokenization ---
    // The selected tokenizer processes the raw text into a sequence of integer IDs.
    let vocab: string[] = [];
    let tokenToIndex: { [key: string]: number } = {};
    let encoded: number[] = [];

    switch(tokenizerType) {
        case 'bpe':
            const { merges, vocab: bpeVocabObj } = trainBPE(textToTrain, vocabSize);
            bpeMergesRef.current = merges;
            bpeVocabRef.current = bpeVocabObj;
            vocab = Object.values(bpeVocabObj) as string[];
            tokenToIndex = Object.fromEntries(Object.entries(bpeVocabObj).map(([id, token]) => [token as string, Number(id)]));
            encoded = encodeBPE(textToTrain, merges);
            break;
        case 'custom':
            let customTokens = customTokenizerSet.split(',').map(t => t.trim()).filter(Boolean);
            const coveredChars = new Set(customTokens.join(''));
            const uniqueCharsInText = new Set(textToTrain.split(''));
            uniqueCharsInText.forEach(char => {
                if (!coveredChars.has(char)) {
                    customTokens.push(char);
                }
            });
            vocab = (Array.from(new Set(customTokens)) as string[]).sort((a: string, b: string) => b.length - a.length || a.localeCompare(b));
            tokenToIndex = Object.fromEntries(vocab.map((token, i) => [token, i]));
            encoded = encodeCustom(textToTrain, vocab, tokenToIndex);
            bpeMergesRef.current = null;
            bpeVocabRef.current = null;
            break;
        case 'character':
        default:
            vocab = (Array.from(new Set(textToTrain.split(''))) as string[]).sort();
            tokenToIndex = Object.fromEntries(vocab.map((char, i) => [char, i]));
            encoded = textToTrain.split('').map(char => tokenToIndex[char]);
            bpeMergesRef.current = null;
            bpeVocabRef.current = null;
            break;
    }
    encodedTextRef.current = encoded;
    
    // --- Model Initialization ---
    // A new model with random weights is created based on the generated vocabulary.
    let newModel: LanguageModel;
    switch(modelType) {
      case 'RNN':
        newModel = initializeRNNModel(vocab, keepParams ? hiddenSize : defaultHiddenSize);
        break;
      case 'GRU':
        newModel = initializeGRUModel(vocab, keepParams ? hiddenSize : defaultHiddenSize);
        break;
      case 'LSTM':
        newModel = initializeLSTMModel(vocab, keepParams ? hiddenSize : defaultHiddenSize);
        break;
      case 'FFNN':
      default:
        newModel = initializeFFNNModel(vocab, keepParams ? hiddenSize : defaultHiddenSize);
        break;
    }
    setModel(newModel);

    // Reset hyperparameters if not explicitly keeping them.
    if (!keepParams) {
        setLearningRate(defaultLearningRate);
        setHiddenSize(defaultHiddenSize);
        setEpochs(defaultEpochs);
        setBatchSize(defaultBatchSize);
        setDropoutRate(0.1);
    }
    setCurrentLearningRate(keepParams ? learningRate : defaultLearningRate);
  }, [modelType, trainingText, tokenizerType, customTokenizerSet, vocabSize, hiddenSize, learningRate, defaultBatchSize, defaultEpochs, defaultHiddenSize, defaultLearningRate]);

  // Effect to re-initialize the model whenever the model type changes.
  useEffect(() => {
    initialize();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelType]);

  // --- TRAINING LOOP ---
  /**
   * Executes a single step of training (one batch/sequence).
   */
  const runTrainingStep = useCallback(() => {
    const currentModel = modelRef.current;
    if (!currentModel) return;

    let result: TrainStepResult;
    switch(modelType) {
      case 'RNN':
        result = trainStepRNN(currentModel as any, encodedTextRef.current, currentStepRef.current, batchSize, currentLearningRate, dropoutRate);
        break;
      case 'GRU':
        result = trainStepGRU(currentModel as any, encodedTextRef.current, currentStepRef.current, batchSize, currentLearningRate, dropoutRate);
        break;
      case 'LSTM':
        result = trainStepLSTM(currentModel as any, encodedTextRef.current, currentStepRef.current, batchSize, currentLearningRate, dropoutRate);
        break;
      case 'FFNN':
      default:
        result = trainStepFFNN(currentModel as any, encodedTextRef.current, currentStepRef.current, batchSize, currentLearningRate);
        break;
    }
    
    modelRef.current = result.updatedModel;
    currentStepRef.current += batchSize;
    currentEpochLossRef.current.push(result.loss);
    
    // Update prediction statistics for the heatmap.
    const stats = predictionStatsRef.current;
    for (const res of result.predictionResults) {
        const { inputToken, targetToken, predictedToken } = res;
        if (!stats[inputToken]) stats[inputToken] = {};
        if (!stats[inputToken][targetToken]) stats[inputToken][targetToken] = { correct: 0, total: 0 };
        stats[inputToken][targetToken].total += 1;
        if (predictedToken === targetToken) {
            stats[inputToken][targetToken].correct += 1;
        }
    }

    // Update UI only if not in fast mode.
    if (!isFastMode) {
      setVisData(result);
      if (currentStepRef.current % (batchSize * 5) === 0) {
        setLogs(prev => [`[E:${currentEpoch}, S:${currentStepRef.current}] Loss: ${result.loss.toFixed(4)}`, ...prev].slice(0, 50));
      }
    }
  }, [modelType, batchSize, currentLearningRate, dropoutRate, isFastMode, currentEpoch]);

  /**
   * Generates a single word for testing or coaching.
   */
  const generateOneWord = useCallback(() => {
    const currentModel = modelRef.current;
    if (!currentModel || !currentModel.vocab || currentModel.vocab.length === 0) return '';
    
    const validStarters = new Set([...ONSETS, ...VOWELS]);
    let seedVocab = currentModel.vocab.filter(token => validStarters.has(token) && token.trim() !== '');
    if (seedVocab.length === 0) {
        seedVocab = currentModel.vocab.filter(c => c.trim() !== '');
    }
    if (seedVocab.length === 0) return '';
    
    const randomIndex = Math.floor(Math.random() * seedVocab.length);
    const seed = seedVocab[randomIndex];
    
    switch(modelType) {
        case 'RNN': return generateRNN(currentModel as any, seed, 50, 1.1);
        case 'GRU': return generateGRU(currentModel as any, seed, 50, 1.1);
        case 'LSTM': return generateLSTM(currentModel as any, seed, 50, 1.1);
        case 'FFNN': default: return generateFFNN(currentModel as any, seed, 50, 1.1);
    }
  }, [modelType]);

  /**
   * The main loop for the auto-coaching feature.
   */
  const autoCoachLoop = useCallback(() => {
    if (!isAutoCoachingRef.current) return;

    const word = generateOneWord();
    if (word) {
        const isGood = isGoodWord(word);
        setAutoCoachedWords(prev => [...prev, { word, isGood }].slice(-100));

        if (isGood) {
            coachingSessionStats.current.good++;
            handleReinforcement(word, false);
            if (!originalTrainingTextRef.current.includes(word)) {
                setNewWordsLog(prev => [...new Set([...prev, word])]); // Ensure unique words
            }
        } else {
            coachingSessionStats.current.bad++;
        }
    }
    
    // Continue the loop.
    setTimeout(autoCoachLoop, 200);
  }, [generateOneWord, handleReinforcement]);

  /**
   * The main training loop, driven by requestAnimationFrame for smooth UI performance.
   */
  const trainingLoop = useCallback(() => {
    if (trainingStateRef.current !== 'RUNNING' && trainingStateRef.current !== 'FINE_TUNING' && trainingStateRef.current !== 'STABILIZED') return;

    runTrainingStep();
    
    // Check if the current epoch is finished.
    if (currentStepRef.current >= encodedTextRef.current.length - 1) {
      const epochLoss = currentEpochLossRef.current.reduce((a, b) => a + b, 0) / currentEpochLossRef.current.length;
      setLossHistory(prev => [...prev, epochLoss]);
      
      // Log epoch summary.
      const prevLoss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : epochLoss;
      const delta = epochLoss - prevLoss;
      const deltaSign = delta >= 0 ? '+' : '';
      setLogs(prev => [`Epoch ${currentEpoch} complete. Loss: ${epochLoss.toFixed(4)} (Δ: ${deltaSign}${delta.toFixed(4)})`, ...prev].slice(0, 50));

      const currentStats = JSON.parse(JSON.stringify(predictionStatsRef.current));
      setHistoricalPredictionStats(prev => [...prev, currentStats]);
      predictionStatsRef.current = {};
      
      setModel(modelRef.current);

      // Generate sample words at intervals.
      if (currentEpoch % 5 === 0 || currentEpoch === 1) {
          const generatedWords = Array.from({length: 5}).map(() => generateOneWord());
          setGenerationHistory(prev => [...prev, { epoch: currentEpoch, words: generatedWords }]);
      }
      
      const nextEpoch = currentEpoch + 1;

      // Logic for cyclical training: switch to COACHING mode.
      if (isCyclicalModeRef.current && trainingStateRef.current === 'RUNNING' && nextEpoch > (currentCycleRef.current * cycleEpochsRef.current) && currentCycleRef.current <= numCyclesRef.current) {
          setTrainingState('COACHING');
          setCurrentEpoch(nextEpoch);
          currentStepRef.current = 0;
          currentEpochLossRef.current = [];
          return; 
      }

      // --- Early Stopping and Fine-Tuning Logic ---
      epochsWithoutImprovementRef.current++;
      if (epochLoss < bestLossRef.current) {
        bestLossRef.current = epochLoss;
        epochsWithoutImprovementRef.current = 0;
      }
      
      const currentState = trainingStateRef.current;
      if (currentState === 'RUNNING' && epochsWithoutImprovementRef.current >= EARLY_STOPPING_PATIENCE) {
          setTrainingState('STABILIZED');
          fineTuningEpochsRemainingRef.current = 5;
          setLogs(prev => [`Loss stabilized. Starting fine-tuning...`, ...prev].slice(0, 50));
      } else if ((currentState === 'STABILIZED' || currentState === 'FINE_TUNING') && fineTuningEpochsRemainingRef.current > 0) {
          setTrainingState('FINE_TUNING');
          // Decay the learning rate for fine-tuning.
          const decayFactor = Math.pow(FINAL_LR_MULTIPLIER, 1/5);
          setCurrentLearningRate(prev => prev * decayFactor);
          fineTuningEpochsRemainingRef.current--;
      } else if (nextEpoch > epochs || (fineTuningEpochsRemainingRef.current <= 0 && currentState === 'FINE_TUNING')) {
          setTrainingState('FINISHED');
          if (isCyclicalModeRef.current) setIsCyclicalMode(false); // Auto-disable checkbox
          if (isFastMode) setIsFastMode(false);
          setLogs(prev => [`Training finished.`, ...prev].slice(0, 50));
          return;
      }

      // Prepare for the next epoch.
      currentStepRef.current = 0;
      currentEpochLossRef.current = [];
      setCurrentEpoch(nextEpoch);
    }

    // Request the next frame to continue the loop.
    loopIdRef.current = requestAnimationFrame(trainingLoop);
  }, [runTrainingStep, epochs, isFastMode, currentEpoch, generateOneWord, lossHistory]);

  // Effect to start/stop the trainingLoop based on the trainingState.
  useEffect(() => {
    if (trainingState === 'RUNNING' || trainingState === 'FINE_TUNING' || trainingState === 'STABILIZED') {
      loopIdRef.current = requestAnimationFrame(trainingLoop);
    } else {
      if (loopIdRef.current) {
        cancelAnimationFrame(loopIdRef.current);
      }
    }
    return () => {
      if (loopIdRef.current) {
        cancelAnimationFrame(loopIdRef.current);
      }
    };
  }, [trainingState, trainingLoop]);

  const coachingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Effect to manage the fixed-duration coaching session in cyclical mode.
  useEffect(() => {
    if (trainingState === 'COACHING') {
      coachingSessionStats.current = { good: 0, bad: 0 };
      setIsAutoCoaching(true);
      
      coachingTimeoutRef.current = setTimeout(() => {
        setIsAutoCoaching(false);
        
        const { good, bad } = coachingSessionStats.current;
        const total = good + bad;
        const ratio = total > 0 ? (good / total * 100).toFixed(1) : '0.0';
        const cycle = currentCycleRef.current;
        const logMessage = `Coaching Cycle ${cycle}: ${good} good, ${bad} bad (${ratio}% success)`;
        setCoachingStatsLog(prev => [...prev, logMessage]);
        setLogs(prev => [logMessage, ...prev].slice(0, 50));

        if (cycle < numCyclesRef.current) {
          setCurrentCycle(prev => prev + 1);
          setTrainingState('RUNNING');
        } else {
          setTrainingState('FINISHED');
          setIsCyclicalMode(false); // Auto-disable checkbox
          setLogs(prev => ['Cyclical training finished.', ...prev].slice(0, 50));
        }

      }, 60000); // 60 seconds
    }

    return () => {
      if (coachingTimeoutRef.current) {
        clearTimeout(coachingTimeoutRef.current);
      }
    };
  }, [trainingState]);

  const handleToggleAutoCoach = () => {
    if (isCyclicalMode) return;
    const nextState = !isAutoCoaching;
    
    if (nextState) { // Turning ON
        if (trainingStateRef.current === 'IDLE') {
            initialize(trainingText, true);
            originalTrainingTextRef.current = trainingText;
            setCoachingEnabled(true);
            setTrainingState('PAUSED'); // Set to paused to indicate model is ready but not training
        }
    
        if (trainingStateRef.current === 'RUNNING') {
            handlePause(); // Pause the main training loop
        }
    }
    
    setIsAutoCoaching(nextState);
  };
  
  // Effect to start the auto-coach loop when its flag is enabled.
  useEffect(() => {
      if (isAutoCoaching) {
          autoCoachLoop();
      }
  }, [isAutoCoaching, autoCoachLoop]);
  
  // --- CONTROLS EVENT HANDLERS ---
  const handleStart = () => {
    if (trainingState === 'IDLE' && !modelIsLoaded) {
      initialize(trainingText, true);
      originalTrainingTextRef.current = trainingText;
    } else if (trainingState === 'FINISHED' && !isCyclicalMode) {
      // Logic to continue training after it has finished.
      setEpochs(prev => prev + 50);
      epochsWithoutImprovementRef.current = 0;
      fineTuningEpochsRemainingRef.current = 0;
      if (lossHistory.length > 0) {
        bestLossRef.current = Math.min(...lossHistory);
      }
    }
    
    if (isCyclicalMode) {
        setEpochs(numCycles * cycleEpochs);
        setCurrentCycle(1);
    }

    setCoachingEnabled(true);
    setTrainingState('RUNNING');

    // Set up the timeout for automatic fast mode.
    if (useAutoFastMode && !isFastMode) {
      fastModeTimeoutRef.current = setTimeout(() => {
        if (trainingStateRef.current === 'RUNNING' && currentEpoch < epochs - 1) {
          setIsFastMode(true);
          setLogs(prev => ['Fast mode enabled to speed up training. UI updates suspended.', ...prev].slice(0, 50));
        }
      }, 60000);
    }
  };
  
  const handleReset = () => {
    initialize();
  };
  
  const handleSaveModel = () => {
    if (!model) return;
    const data = {
      model,
      modelType,
      tokenizerInfo: {
        type: tokenizerType,
        bpeMerges: tokenizerType === 'bpe' && bpeMergesRef.current ? Array.from(bpeMergesRef.current.entries()) : null,
        bpeVocab: tokenizerType === 'bpe' ? bpeVocabRef.current : null,
        customSet: tokenizerType === 'custom' ? customTokenizerSet : null,
      }
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${modelType}-model.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  const handleLoadModel = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const data = JSON.parse(event.target?.result as string);
          if (data.model && data.modelType && data.tokenizerInfo) {
            if(data.modelType !== modelType) {
              alert(`This file contains a ${data.modelType} model. Please switch to the correct tab to load it.`);
              return;
            }
            setModel(data.model);
            setTokenizerType(data.tokenizerInfo.type);
            
            if(data.tokenizerInfo.type === 'bpe' && data.tokenizerInfo.bpeMerges && data.tokenizerInfo.bpeVocab) {
              bpeMergesRef.current = new Map(data.tokenizerInfo.bpeMerges);
              bpeVocabRef.current = data.tokenizerInfo.bpeVocab;
              const encoded = encodeBPE(trainingText, bpeMergesRef.current);
              encodedTextRef.current = encoded;
            } else if (data.tokenizerInfo.type === 'custom' && data.tokenizerInfo.customSet) {
                setCustomTokenizerSet(data.tokenizerInfo.customSet);
                const vocab = data.model.vocab;
                const tokenToIndex = data.model.tokenToIndex;
                encodedTextRef.current = encodeCustom(trainingText, vocab, tokenToIndex);
            }

            setModelIsLoaded(true);
            setTrainingState('IDLE');
            setLogs(['Model loaded successfully. Ready to train.']);
            setVisData(null);
            setLossHistory([]);
            setGenerationHistory([]);
            setCurrentEpoch(1);
            setCoachingEnabled(false);
          }
        } catch (error) {
          console.error("Failed to load model:", error);
          alert("Failed to load or parse model file.");
        }
      };
      reader.readAsText(file);
    };
    input.click();
  };

  const handleDownloadNewWords = () => {
    const textToSave = newWordsLog.join('\n');
    const blob = new Blob([textToSave], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'discovered-words.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  /**
   * Toggles the accordion sections in the sidebar.
   * @param {AccordionTab} tab - The tab to open or close.
   */
  const toggleAccordion = (tab: AccordionTab) => {
    setOpenAccordion(openAccordion === tab ? null : tab);
  };

  const getIntroParagraph = () => {
    switch (modelType) {
      case 'FFNN':
        return "This playground trains a Feed-Forward Neural Network (FFNN), the most fundamental neural network architecture. The key feature of an FFNN is that it's memoryless. It predicts the next token based only on the single, immediate input token, making it great for learning simple, direct relationships but unable to grasp longer-term context.";
      case 'RNN':
        return "This tab introduces a Recurrent Neural Network (RNN). This is a significant step up from the FFNN because it has memory. An RNN maintains an internal 'hidden state' that acts as a summary of the sequence it has seen so far. This allows it to make more context-aware predictions, as it considers both the current input and its memory of previous inputs.";
      case 'GRU':
        return "Here we train a Gated Recurrent Unit (GRU), an advanced RNN designed to better handle long-range dependencies. A GRU uses intelligent 'gates' (an update gate and a reset gate) that learn to control the flow of information. This allows the model to selectively forget irrelevant past information and update its memory with what's important, leading to a more stable and powerful long-term memory.";
      case 'LSTM':
        return "This playground features a Long Short-Term Memory (LSTM) network, one of the most powerful and widely used recurrent architectures. The LSTM introduces a separate 'cell state' for long-term memory, which is carefully managed by three distinct gates: a forget gate, an input gate, and an output gate. This sophisticated mechanism allows LSTMs to track multiple pieces of information over very long sequences, making them exceptionally good at capturing complex patterns in text.";
      default:
        return "";
    }
  };

  return (
    <div className="space-y-6">
      {/* --- Intro Paragraph --- */}
      <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
        <p className="text-gray-400">
          {getIntroParagraph()}
        </p>
      </div>
      {/* --- Main Controls --- */}
      <Controls
        trainingState={trainingState}
        onStart={handleStart}
        onPause={handlePause}
        onReset={handleReset}
        onSaveModel={handleSaveModel}
        onLoadModel={handleLoadModel}
        model={model}
        trainingText={trainingText} setTrainingText={setTrainingText}
        learningRate={learningRate} setLearningRate={setLearningRate}
        hiddenSize={hiddenSize} setHiddenSize={setHiddenSize}
        epochs={isCyclicalMode ? numCycles * cycleEpochs : epochs} setEpochs={setEpochs}
        batchSize={batchSize} setBatchSize={setBatchSize}
        tokenizerType={tokenizerType} setTokenizerType={setTokenizerType}
        customTokenizerSet={customTokenizerSet} setCustomTokenizerSet={setCustomTokenizerSet}
        vocabSize={vocabSize} setVocabSize={setVocabSize}
        dropoutRate={dropoutRate} setDropoutRate={setDropoutRate}
        useAutoFastMode={useAutoFastMode} setUseAutoFastMode={setUseAutoFastMode}
        batchSizeLabel={batchSizeLabel}
        batchSizeTooltip={batchSizeTooltip}
        modelType={modelType}
        isAutoCoaching={isAutoCoaching}
        isCyclicalMode={isCyclicalMode} setIsCyclicalMode={setIsCyclicalMode}
        cycleEpochs={cycleEpochs} setCycleEpochs={setCycleEpochs}
        numCycles={numCycles} setNumCycles={setNumCycles}
        currentCycle={currentCycle}
      />
      {/* --- Main Layout: Visualizations and Sidebar --- */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <ArchitectureVisualizer model={model} visData={visData} isFastMode={isFastMode} currentEpoch={currentEpoch} />
          <LossHistogram lossHistory={lossHistory} />
          <SuccessRateHeatmap statsHistory={historicalPredictionStats} vocab={model?.vocab || []} />
        </div>
        <div className="space-y-4 lg:sticky lg:top-6 self-start">
            <AccordionItem title="Training Log" isOpen={openAccordion === 'log'} onToggle={() => toggleAccordion('log')}>
                <LogPanel
                    logs={logs}
                    epoch={currentEpoch}
                    totalEpochs={isCyclicalMode ? numCycles * cycleEpochs : epochs}
                    loss={visData?.loss ?? null}
                    currentLearningRate={currentLearningRate}
                    isFastMode={isFastMode}
                />
            </AccordionItem>
            <AccordionItem title="Generate & Coach" isOpen={openAccordion === 'generate'} onToggle={() => toggleAccordion('generate')}>
                <GenerationPanel 
                    model={model} 
                    coachingEnabled={coachingEnabled} 
                    onReinforce={handleReinforcement}
                    isAutoCoaching={isAutoCoaching}
                    onToggleAutoCoach={handleToggleAutoCoach}
                    autoCoachedWords={autoCoachedWords}
                    newWordsLog={newWordsLog}
                    coachingStatsLog={coachingStatsLog}
                    onDownloadNewWords={handleDownloadNewWords}
                />
            </AccordionItem>
            <AccordionItem title="Generation History" isOpen={openAccordion === 'history'} onToggle={() => toggleAccordion('history')}>
                <GenerationHistoryPanel history={generationHistory} />
            </AccordionItem>
        </div>
      </div>
    </div>
  );
}