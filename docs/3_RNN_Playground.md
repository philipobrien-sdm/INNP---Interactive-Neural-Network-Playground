
# Understanding the RNN Playground

## 1. High-Level Overview

The **RNN** tab introduces a **Recurrent Neural Network**. This is a significant step up from the FFNN because it introduces the concept of **memory**.

An RNN processes sequences of data. It maintains an internal "hidden state" that acts as a summary of the information it has seen so far in the sequence. When it makes a prediction, it uses both the current input token and its own hidden state (memory) from the previous step.

**Key Concept:** The RNN's recurrent connection (`Whh` weights) allows it to learn patterns that span multiple time steps. When predicting the character after `"p"` in `"splonder"`, its hidden state contains information about having just seen `"s"` and `"l"`, allowing for a more context-aware prediction than the FFNN.

---

## 2. Key Components & Files

-   **`components/AdvancedPlayground.tsx`**: A wrapper component that renders `Playground.tsx` with the configuration for an RNN (`modelType="RNN"`). It also changes the "Batch Size" label to "Sequence Length", as this is more appropriate for RNNs.
-   **`components/Playground.tsx`**: The central component managing the training lifecycle. The core logic remains the same, but it will now call the RNN-specific functions.
-   **`services/languageModel.ts`**: Contains the from-scratch implementation of the RNN:
    -   `initializeRNNModel`: Creates an RNN with its unique weight matrices.
    -   `trainStepRNN`: Implements the forward and backward passes for an RNN.
    -   `generateRNN`: Generates text using the trained RNN model.

---

## 3. The Execution Flow: What's Different?

The overall lifecycle (Initialize -> Start -> Loop -> Finish) is identical to the FFNN and is managed by `Playground.tsx`. The crucial differences lie within the model-specific functions called from `services/languageModel.ts`.

### Step 1: Initialization

-   The `initialize` function in `Playground.tsx` now calls `initializeRNNModel(vocab, hiddenSize)`.
-   **Inside `initializeRNNModel`**:
    -   Instead of simple `hiddenLayer` and `outputLayer`, it creates three distinct layers:
        1.  **`Wxh`**: Weights connecting the **I**nput (`x`) to the **H**idden state (`h`).
        2.  **`Whh`**: Weights connecting the previous **H**idden state to the new **H**idden state. **This is the recurrent connection—the "memory loop".**
        3.  **`Why`**: Weights connecting the **H**idden state to the **O**utput (`y`).
    -   It also initializes the hidden state `h`, a vector of zeros that will be updated at each step.

### Step 2: The Training Step

-   The `runTrainingStep` function in `Playground.tsx` now calls:
    ```javascript
    // in Playground.tsx -> runTrainingStep()
    result = trainStepRNN(currentModel, encodedText, currentStep, sequenceLength, learningRate, dropoutRate);
    ```
-   **Inside `trainStepRNN`**: This function is more complex than the FFNN's because it must process a whole `sequenceLength` of tokens. This process is called **Backpropagation Through Time (BPTT)**.
    -   **Forward Pass (Through Time)**:
        1.  The function loops from `t=0` to `sequenceLength - 1`.
        2.  At each time step `t`, it calculates the new hidden state using the formula: `h_t = tanh(x_t · Wxh + h_{t-1} · Whh + biases)`. Notice how the previous hidden state `h_{t-1}` is part of the calculation.
        3.  It then calculates the output probabilities for that step: `probs_t = softmax(h_t · Why + biases)`.
        4.  The loss for step `t` is calculated, and all intermediate activations (`h_t`, `probs_t`, etc.) are stored in a `cache`.
        5.  The final `h_t` becomes `h_{t-1}` for the next iteration.

    -   **Backward Pass (Through Time)**:
        1.  After the forward pass is complete, the function loops **backwards** from `t = sequenceLength - 1` down to `0`.
        2.  At each step `t`, it calculates the gradients for that step using the values stored in the `cache`.
        3.  Crucially, the gradient of the hidden state (`dh`) is also propagated backward in time. The error from step `t` influences the gradient calculation at step `t-1`. This is how the model learns dependencies across the sequence.
        4.  The gradients for all weight matrices (`dWxh`, `dWhh`, `dWhy`) are accumulated over the entire backward pass.

    -   **Weight Update**:
        1.  After the backward pass is complete, the accumulated gradients are used to update the model's weights, just like in the FFNN.
        2.  The final hidden state of the sequence is saved back to the model object (`updatedModel.h`) to be used as the starting memory for the next training sequence.

### Step 3: Generation

-   The `generateRNN` function is called for text generation.
-   **How it Works**:
    1.  It starts with the model's current hidden state `h`.
    2.  It takes a `seed` character and performs a single forward pass step to generate the next character's probabilities and a new hidden state `h_new`.
    3.  It samples the next character from the probabilities.
    4.  This new character becomes the input for the next step, and `h_new` is used as the previous hidden state.
    5.  This loop continues, constantly updating the hidden state, which allows the generated text to have a basic level of coherence.
