
import React from 'react';
import { Playground } from './Playground';

export const AdvancedPlayground = () => {
  return (
    <Playground
      modelType="RNN"
      batchSizeLabel="Sequence Length"
      batchSizeTooltip="The number of consecutive tokens the RNN processes in one training step (backpropagation through time). Longer sequences help the model learn long-term dependencies but are computationally more expensive."
      defaultLearningRate={0.005}
      defaultHiddenSize={48}
      defaultEpochs={100}
      defaultBatchSize={32}
    />
  );
};