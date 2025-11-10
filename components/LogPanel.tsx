/**
 * @file LogPanel.tsx
 * @description A component that displays the real-time progress and logs of the training process.
 * It provides at-a-glance information about the current epoch, loss, and learning rate,
 * as well as a detailed log of individual training steps.
 */

import React from 'react';
import { Tooltip } from './Tooltip.js';

/**
 * Renders the primary logging and progress-tracking interface.
 * @param {object} props - The component's props.
 * @param {string[]} props.logs - An array of log messages to display.
 * @param {number} props.epoch - The current training epoch.
 * @param {number} props.totalEpochs - The total number of epochs planned.
 * @param {number | null} props.loss - The loss value from the latest training step.
 * @param {number} props.currentLearningRate - The effective learning rate for the current step.
 * @param {boolean} props.isFastMode - A flag indicating if UI updates are suspended.
 */
export const LogPanel = ({ logs, epoch, totalEpochs, loss, currentLearningRate, isFastMode }) => {
  // Calculate completed epochs for the progress bar.
  const completedEpochs = Math.max(0, epoch - 1);
  return (
    <div className="flex flex-col">
      {/* --- Epoch Progress Bar and Stats --- */}
      <Tooltip text="Tracks the overall progress through all epochs. Loss measures how 'wrong' the model's prediction is for the current step. The goal is to minimize the loss.">
        <div className="bg-gray-900 p-3 rounded-md mb-4 flex-shrink-0 border border-gray-700">
          <div className="flex justify-between text-sm flex-wrap">
            <span className="font-mono mr-4">Epoch: {epoch} / {totalEpochs}</span>
            <span className="font-mono mr-4">Loss: {loss !== null ? loss.toFixed(4) : 'N/A'}</span>
            {/* Display learning rate, which can change during fine-tuning. */}
            {currentLearningRate !== null && currentLearningRate !== undefined && (
                <span className="font-mono">LR: {currentLearningRate.toExponential(2)}</span>
            )}
          </div>
          {/* Visual progress bar for the entire training run. */}
          <div className="w-full bg-gray-700 rounded-full h-2.5 mt-2">
              <div className="bg-cyan-500 h-2.5 rounded-full" style={{ width: `${(completedEpochs / totalEpochs) * 100}%` }}></div>
          </div>
        </div>
      </Tooltip>

      {/* --- Real-time Log Feed --- */}
      <Tooltip text="A real-time feed of the training process. Each line shows the model processing one character, predicting the next, and the resulting loss.">
        <div className="h-64 overflow-y-auto bg-gray-900 p-3 rounded-md border border-gray-700">
          <ul className="text-xs font-mono text-gray-400 space-y-1">
            {/* Display a sticky message at the top when fast mode is active. */}
            {isFastMode && (
                <li className="sticky top-0 bg-gray-900 py-1 text-yellow-400 font-bold z-10 animate-pulse">
                    <span className="text-gray-600 mr-2">{'>'}</span>Fast Mode Active - UI updates suspended...
                </li>
            )}
            {/* Render the list of log messages. */}
            {logs.map((log, i) => (
              <li key={i} className={`whitespace-pre-wrap ${i === 0 ? 'text-gray-100 animate-pulse-fast' : ''}`}>
                <span className="text-gray-600 mr-2">{'>'}</span>{log}
              </li>
            ))}
          </ul>
        </div>
      </Tooltip>
    </div>
  );
};
