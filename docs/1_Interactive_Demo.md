
# Understanding the Interactive Demo

## 1. High-Level Overview

The **Interactive Demo** is the best place to start. Its purpose is to provide a slow, clear, and detailed visualization of a single training step for the simplest model, the **Feed-Forward Neural Network (FFNN)**.

Unlike the other tabs which run through thousands of training steps quickly, this demo focuses on a tiny, predictable dataset (`"abcabc"`) and lets you walk through the process one step at a time. It explicitly shows the matrix math involved in making a prediction (the **forward pass**) and learning from a mistake (the **backward pass**).

**Key Concept:** An FFNN has no memory. It makes a prediction for the next character based *only* on the single character it's currently looking at.

---

## 2. Key Components & Files

-   **`components/InteractiveDemo.tsx`**: The main React component that manages the entire state and UI for this tab. It contains the logic for stepping through the training process, rendering the visualizations, and updating the log.
-   **`services/languageModel.ts`**: This file contains the core neural network logic. The demo primarily calls the `initializeFFNNModel` and `trainStepFFNN` functions.
-   **`components/LineArchitectureVisualizer.tsx`**: A specific visualizer for this demo that shows neurons and the connections between them.

---

## 3. The Execution Flow: A Single Step

Here's what happens when you click the **"Next Step"** button:

1.  **Function Call**: The `onClick` handler calls the `handleRunStep` function inside `InteractiveDemo.tsx`.

2.  **State Check**: `handleRunStep` checks if the current epoch is finished. If it is, it calculates the average loss for that epoch, resets the step counter, and increments the epoch number.

3.  **Core Training Logic**: The most important call is made:
    ```javascript
    // in InteractiveDemo.tsx -> handleRunStep()
    const result = trainStepFFNN(model, encodedText, currentStep, 1, INTERACTIVE_LR);
    ```
    This function, located in `services/languageModel.ts`, performs one full training cycle on a single input/target pair (e.g., input `'a'`, target `'b'`).

4.  **Inside `trainStepFFNN`**:
    -   **Forward Pass (Prediction)**:
        1.  The input character (`'a'`) is converted into a one-hot vector (e.g., `[1, 0, 0]`).
        2.  **Hidden Layer Activation**: The input vector is multiplied by the hidden layer's weights and a bias is added. This result is passed through a `tanh` activation function. The formula is `h = tanh(input · W_hidden + b_hidden)`.
        3.  **Output Logits**: The hidden activation is then multiplied by the output layer's weights and a bias is added. This produces raw scores, called "logits". The formula is `logits = h · W_output + b_output`.
        4.  **Output Probabilities**: The `softmax` function is applied to the logits, converting them into a probability distribution (a vector of positive numbers that sum to 1). The highest probability corresponds to the model's prediction.
        5.  **Loss Calculation**: The "Cross-Entropy Loss" is calculated using the formula `-log(probability of the correct target)`. A high loss means the model was very "surprised" and incorrect; a low loss means it was confident and correct.

    -   **Backward Pass (Learning)**:
        1.  The function now calculates **gradients**. A gradient is a vector that points in the direction of the steepest increase of the loss. In simple terms, it tells us how much each weight and bias contributed to the final error.
        2.  The gradient calculation starts at the output and works its way backward through the network, using the chain rule from calculus. This process is called **backpropagation**.
        3.  It calculates the gradients for the output layer's weights and biases, and then for the hidden layer's weights and biases.

    -   **Weight Update**:
        1.  The function updates every weight and bias in the model using the formula: `new_weight = old_weight - learning_rate * gradient`.
        2.  By subtracting a small fraction of the gradient, we nudge the weights in the direction that will reduce the loss for this specific example.

5.  **Return Value**: `trainStepFFNN` returns a `result` object containing the updated model, the loss, and all the intermediate matrices (activations, logits, gradients) calculated during the step.

6.  **State Update & Re-render**: Back in `InteractiveDemo.tsx`, the component's state is updated with the new data:
    ```javascript
    // in InteractiveDemo.tsx -> handleRunStep()
    setModel(result.updatedModel);
    setVisData(result);
    // ... and other state updates for step, epoch, and log
    ```
    This `setState` call triggers React to re-render the entire component, displaying the new visualizations and log entries to the user.

---

## 4. Understanding the Visualizations

-   **Line Architecture Visualizer**: Shows the neurons in each layer. The brightness of a neuron indicates its activation level (brighter = more activated). The thickness and color of the lines between neurons represent the connection weights.
-   **Step-by-Step Calculation Breakdown**: This is the core of the demo. It renders each matrix involved in the forward and backward pass using the `MatrixDisplay` component, allowing you to see the exact numbers that the model is working with.
-   **Log**: Displays a running commentary of the training process, showing the input, target, prediction, and loss for each step.
-   **Loss Histogram**: Tracks the average loss at the end of each epoch, giving you a high-level view of whether the model is learning over time.
