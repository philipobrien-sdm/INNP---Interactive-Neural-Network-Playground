
import React from 'react';
import { Playground } from './Playground';

export const LSTMPlayground = () => {
  return (
    <Playground
      modelType="LSTM"
      batchSizeLabel="Sequence Length"
      batchSizeTooltip="The number of consecutive tokens the LSTM processes in one training step. The LSTM's cell state and gating mechanism are specifically designed to capture very long-term dependencies in data."
      defaultLearningRate={0.008}
      defaultHiddenSize={64}
      defaultEpochs={100}
      defaultBatchSize={56}
    />
  );
};