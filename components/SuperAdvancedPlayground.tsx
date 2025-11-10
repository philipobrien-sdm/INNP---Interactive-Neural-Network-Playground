
import React from 'react';
import { Playground } from './Playground';

export const SuperAdvancedPlayground = () => {
  return (
    <Playground
      modelType="GRU"
      batchSizeLabel="Sequence Length"
      batchSizeTooltip="The number of consecutive tokens the GRU processes in one training step. The GRU's gating mechanism is designed to handle long-term dependencies more effectively than a simple RNN."
      defaultLearningRate={0.007}
      defaultHiddenSize={56}
      defaultEpochs={100}
      defaultBatchSize={48}
    />
  );
};