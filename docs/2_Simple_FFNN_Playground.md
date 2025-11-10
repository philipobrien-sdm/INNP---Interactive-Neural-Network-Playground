
# Understanding the Simple (FFNN) Playground

## 1. High-Level Overview

The **Simple (FFNN)** tab allows you to train a **Feed-Forward Neural Network** on a larger, more complex dataset. This is the most fundamental type of neural network architecture in the app.

The primary goal of this tab is to demonstrate the core concepts of tokenization, training loops, hyperparameter tuning, and model evaluation without the added complexity of memory or recurrent connections.

**Key Concept:** The FFNN is **memoryless**. When predicting the character that follows `"p"` in `"splonder"`, it has no knowledge that `"s"` and `"l"` came before. It only sees the `"p"`. This makes it good at learning simple token-to-token relationships but poor at understanding longer contexts.

---

## 2. Key Components & Files

-   **`components/SimplePlayground.tsx`**: This is a simple wrapper component. Its only job is to render the main `Playground` component and pass it the specific configuration for an FFNN (like `modelType="FFNN"`, default hyperparameters, etc.).
-   **`components/Playground.tsx`**: This is the "brain" of the application. It manages all the state, logic, and UI orchestration for the training process. It is designed to be reusable for different model types.
-   **`services/languageModel.ts`**: Contains the from-scratch implementation of the FFNN, including initialization (`initializeFFNNModel`), the training step (`trainStepFFNN`), and text generation (`generateFFNN`).

---

## 3. The Execution Flow: From Start to Finish

### Step 1: Initialization

When the tab first loads or when you click the **"Reset"** button, the `initialize` function in `Playground.tsx` is called. It performs the following actions:

1.  **Resets State**: Clears all logs, visualizations, and resets epoch/step counters.
2.  **Tokenization**: It processes the text from the "Training Text" area based on the selected tokenizer.
    -   **Character**: Creates a vocabulary of every unique character.
    -   **BPE**: Calls `trainBPE` from `services/bpe.ts` to learn a sub-word vocabulary.
    -   **Custom**: Uses the user-provided list of tokens.
    The result is a vocabulary (`vocab`), a mapping from tokens to integers (`tokenToIndex`), and the entire training text converted into a long array of integers (`encodedText`).
3.  **Model Creation**: It calls `initializeFFNNModel(vocab, hiddenSize)` from `services/languageModel.ts`. This function creates the model's structure:
    -   It creates a `hiddenLayer` and an `outputLayer`.
    -   Each layer is initialized with small, random weight values and zeroed biases. This random starting point is crucial for the learning process.
4.  The newly created model object is saved to the component's state using `setModel()`.

### Step 2: Starting the Training

1.  **User Action**: You click the **"Start Training"** button.
2.  **Function Call**: This triggers the `handleStart` function in `Playground.tsx`.
3.  **State Change**: `handleStart` sets the `trainingState` to `'RUNNING'`.
4.  **Loop Begins**: A `useEffect` hook, which listens for changes to `trainingState`, sees the new `'RUNNING'` state and kicks off the `trainingLoop` using `requestAnimationFrame`. Using `requestAnimationFrame` ensures the high-frequency training loop doesn't block the browser's UI, keeping the app responsive.

### Step 3: The Training Loop

The `trainingLoop` function is the heart of the process. On every frame, it does the following:

1.  **Function Call**: It calls `runTrainingStep()`.
2.  **Inside `runTrainingStep`**: This function calls the model-specific training logic:
    ```javascript
    // in Playground.tsx -> runTrainingStep()
    result = trainStepFFNN(currentModel, encodedText, currentStep, batchSize, learningRate);
    ```
    This is the exact same function used in the Interactive Demo, but now it processes a `batchSize` of examples instead of just one. It performs the forward pass, backward pass, and weight update, returning the updated model and training metrics.
3.  **Update Model**: The `modelRef.current` is updated with `result.updatedModel`. Using a `ref` here is a performance optimization to avoid re-rendering the entire app on every single training step.
4.  **Update UI (Conditionally)**: If "Fast Mode" is not active, `runTrainingStep` calls `setVisData(result)` and `setLogs(...)` to update the visualizations and log panel.
5.  **Epoch Management**: The `trainingLoop` checks if the `currentStep` has reached the end of the `encodedText`. If so:
    -   An epoch is complete. The average loss for the epoch is calculated and added to `lossHistory`.
    -   Sample words are generated using `generateFFNN` and added to the `generationHistory`.
    -   **Early stopping logic** is checked. If the loss hasn't improved for a set number of epochs, the training state may change to `'FINE_TUNING'`, which reduces the learning rate.
    -   The `currentEpoch` is incremented and `currentStep` is reset to 0.
6.  **Loop Continuation**: The loop requests the next animation frame to continue the process. This continues until the state changes to `'PAUSED'` or `'FINISHED'`.

### Step 4: Generation & Coaching

1.  **User Action**: You click **"Generate Word"** in the "Generate & Coach" panel.
2.  **Function Call**: This calls `handleGenerate` in `GenerationPanel.tsx`, which in turn calls `generateFFNN` from `services/languageModel.ts`.
3.  **How `generateFFNN` Works**:
    -   It takes a `seed` character and performs a forward pass to get the probabilities for the next character.
    -   It uses the `temperature` setting to adjust the probability distribution. Higher temperature makes the output more random.
    -   It *samples* from this distribution to pick the next character.
    -   This newly generated character becomes the input for the next step. This repeats until a space is generated or a max length is reached.
4.  **User Action**: You click the **"Good 👍"** button.
5.  **Function Call**: This triggers the `handleReinforcement` function in `Playground.tsx`.
6.  **Reinforcement Logic**: This function runs a mini-training loop, calling `trainStepFFNN` repeatedly only on the sequence of characters that formed the "good" word. This strengthens the specific neural pathways that led to that successful output.
