
import React from 'react';
import { Playground } from './Playground';

export const SimplePlayground = () => {
  return (
    <Playground
      modelType="FFNN"
      batchSizeLabel="Batch Size"
      batchSizeTooltip="The number of input-target pairs to process before updating the model's weights. A larger batch size provides a more stable gradient but uses more memory."
      defaultLearningRate={0.01}
      defaultHiddenSize={64}
      defaultEpochs={100}
      defaultBatchSize={64}
    />
  );
};