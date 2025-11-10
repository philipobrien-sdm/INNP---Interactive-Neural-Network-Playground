/**
 * @file languageModel.ts
 * @description This file contains the core implementation of the neural networks from scratch.
 * It includes matrix math utilities, activation functions, and the full forward pass,
 * backward pass (backpropagation), and weight update logic for four different architectures:
 * FFNN, RNN, GRU, and LSTM. It also includes the text generation logic for each model.
 */

import { FFNNModel, GRULanguageModel, Layer, Matrix, RNNModel, LSTMLanguageModel, TrainStepResult } from '../types';

// --- Matrix Math Utilities ---
// These are fundamental building blocks for all neural network operations.

/**
 * Creates a new matrix of specified dimensions.
 * @param rows - The number of rows.
 * @param cols - The number of columns.
 * @param fill - A value or function to initialize each element. Defaults to 0.
 * @returns A new Matrix object.
 */
const createMatrix = (rows: number, cols: number, fill: number | (() => number) = 0): Matrix => {
    const data = Array(rows).fill(0).map(() => 
        Array(cols).fill(0).map(() => (typeof fill === 'function' ? fill() : fill))
    );
    return { rows, cols, data };
};

/**
 * Performs matrix multiplication (dot product) between two matrices.
 * @param a - The first matrix.
 * @param b - The second matrix.
 * @returns The resulting matrix.
 */
const dot = (a: Matrix, b: Matrix): Matrix => {
    if (a.cols !== b.rows) throw new Error("Matrix dimension mismatch for dot product.");
    const result = createMatrix(a.rows, b.cols);
    for (let i = 0; i < a.rows; i++) {
        for (let j = 0; j < b.cols; j++) {
            let sum = 0;
            for (let k = 0; k < a.cols; k++) {
                sum += a.data[i][k] * b.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    return result;
};

/**
 * Performs element-wise addition of two matrices.
 * @param a - The first matrix.
 * @param b - The second matrix.
 * @returns The resulting matrix.
 */
const add = (a: Matrix, b: Matrix): Matrix => {
    if (a.rows !== b.rows || a.cols !== b.cols) {
        // Biases are often 1xN vectors added to MxN matrices. This handles that "broadcasting".
        if (a.rows > 1 && b.rows === 1) {
            const result = createMatrix(a.rows, a.cols);
            for (let i = 0; i < a.rows; i++) {
                for (let j = 0; j < a.cols; j++) {
                    result.data[i][j] = a.data[i][j] + b.data[0][j];
                }
            }
            return result;
        }
        console.error("Matrix dimension mismatch for addition.", a, b);
        return a;
    }
    const result = createMatrix(a.rows, a.cols);
    for (let i = 0; i < a.rows; i++) {
        for (let j = 0; j < a.cols; j++) {
            result.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }
    return result;
};

/**
 * Performs element-wise subtraction of two matrices.
 * @param a - The first matrix.
 * @param b - The second matrix.
 * @returns The resulting matrix.
 */
const subtract = (a: Matrix, b: Matrix): Matrix => {
    if (a.rows !== b.rows || a.cols !== b.cols) throw new Error("Matrix dimension mismatch for subtraction.");
    const result = createMatrix(a.rows, a.cols);
    for (let i = 0; i < a.rows; i++) {
        for (let j = 0; j < a.cols; j++) {
            result.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }
    return result;
};

/**
 * Performs element-wise multiplication (Hadamard product) of two matrices.
 * @param a - The first matrix.
 * @param b - The second matrix.
 * @returns The resulting matrix.
 */
const multiply = (a: Matrix, b: Matrix): Matrix => {
    if (a.rows !== b.rows || a.cols !== b.cols) throw new Error("Matrix dimension mismatch for element-wise multiplication.");
    const result = createMatrix(a.rows, a.cols);
    for (let i = 0; i < a.rows; i++) {
        for (let j = 0; j < a.cols; j++) {
            result.data[i][j] = a.data[i][j] * b.data[i][j];
        }
    }
    return result;
};

/**
 * Multiplies every element of a matrix by a scalar value.
 * @param m - The matrix.
 * @param s - The scalar.
 * @returns The resulting matrix.
 */
const scale = (m: Matrix, s: number): Matrix => {
    const result = createMatrix(m.rows, m.cols);
    for (let i = 0; i < m.rows; i++) {
        for (let j = 0; j < m.cols; j++) {
            result.data[i][j] = m.data[i][j] * s;
        }
    }
    return result;
};

/**
 * Transposes a matrix (swaps rows and columns).
 * @param m - The matrix.
 * @returns The transposed matrix.
 */
const transpose = (m: Matrix): Matrix => {
    const result = createMatrix(m.cols, m.rows);
    for (let i = 0; i < m.rows; i++) {
        for (let j = 0; j < m.cols; j++) {
            result.data[j][i] = m.data[i][j];
        }
    }
    return result;
};

/**
 * Applies a function to every element of a matrix.
 * @param m - The matrix.
 * @param fn - The function to apply.
 * @returns The resulting matrix.
 */
const map = (m: Matrix, fn: (val: number) => number): Matrix => {
    const result = createMatrix(m.rows, m.cols);
    for (let i = 0; i < m.rows; i++) {
        for (let j = 0; j < m.cols; j++) {
            result.data[i][j] = fn(m.data[i][j]);
        }
    }
    return result;
};

/**
 * Creates a matrix filled with ones.
 * @param rows - The number of rows.
 * @param cols - The number of columns.
 * @returns A new matrix of ones.
 */
const ones = (rows: number, cols: number) => createMatrix(rows, cols, 1);

// --- Activation Functions ---
// These non-linear functions are applied to neuron outputs to allow the network to learn complex patterns.

/** Hyperbolic Tangent activation function. Squashes values to be between -1 and 1. */
const tanh = (x: number): number => Math.tanh(x);
/** Derivative of tanh. Needed for backpropagation. `y` is the output of tanh(x). */
const dtanh = (y: number): number => 1 - y * y;

/** Sigmoid activation function. Squashes values to be between 0 and 1. Often used in gates (GRU, LSTM). */
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
/** Derivative of sigmoid. Needed for backpropagation. `y` is the output of sigmoid(x). */
const dsigmoid = (y: number): number => y * (1 - y);

/**
 * Softmax function. Converts a vector of raw scores (logits) into a probability distribution.
 * The output values are all positive and sum to 1.
 * @param m - A 1xN matrix of logits.
 * @returns A 1xN matrix of probabilities.
 */
const softmax = (m: Matrix): Matrix => {
    if (m.data.length === 0 || m.data[0].length === 0) return createMatrix(1, m.cols);
    // Subtracting the max value is a trick for numerical stability to prevent Math.exp() from overflowing.
    const maxVal = Math.max(...m.data[0]);
    const exps = m.data[0].map(x => Math.exp(x - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const result = createMatrix(1, m.cols);
    result.data[0] = exps.map(e => e / (sumExps || 1)); // The '|| 1' prevents division by zero.
    return result;
};

// --- FFNN Implementation ---
/**
 * Creates a single layer with randomly initialized weights and zeroed biases.
 * @param inputSize - Number of neurons in the previous layer.
 * @param outputSize - Number of neurons in this layer.
 * @returns A new Layer object.
 */
const createLayer = (inputSize: number, outputSize: number): Layer => ({
    // Weights are initialized with small random values to break symmetry and start the learning process.
    weights: createMatrix(inputSize, outputSize, () => (Math.random() - 0.5) * 0.1),
    biases: createMatrix(1, outputSize),
});

/**
 * Initializes a new Feed-Forward Neural Network model.
 * @param vocab - The list of unique tokens.
 * @param hiddenSize - The number of neurons in the hidden layer.
 * @returns A new FFNNModel object.
 */
export const initializeFFNNModel = (vocab: string[], hiddenSize: number): FFNNModel => {
    const vocabSize = vocab.length;
    const tokenToIndex = Object.fromEntries(vocab.map((char, i) => [char, i]));
    return {
        type: 'FFNN',
        vocab,
        tokenToIndex,
        hiddenLayer: createLayer(vocabSize, hiddenSize),
        outputLayer: createLayer(hiddenSize, vocabSize),
    };
};

/**
 * Performs a single training step (forward pass, loss calculation, backward pass, and weight update) for an FFNN.
 * This function processes a batch of input-target pairs.
 */
export const trainStepFFNN = (
    model: FFNNModel,
    encodedText: number[],
    step: number,
    batchSize: number,
    learningRate: number
): TrainStepResult => {
    const vocabSize = model.vocab.length;
    const hiddenSize = model.hiddenLayer.weights.cols;
    
    // Initialize gradients for this batch to all zeros.
    let hiddenGrad = { weights: createMatrix(vocabSize, hiddenSize), biases: createMatrix(1, hiddenSize) };
    let outputGrad = { weights: createMatrix(hiddenSize, vocabSize), biases: createMatrix(1, vocabSize) };
    
    let totalLoss = 0;
    let lastResultForVis: any = {};
    const predictionResults: { inputToken: string; targetToken: string; predictedToken: string; }[] = [];

    // Process each item in the batch.
    const batchEnd = Math.min(step + batchSize, encodedText.length - 1);
    for (let i = step; i < batchEnd; i++) {
        const inputIndex = encodedText[i];
        const targetIndex = encodedText[i + 1];

        // --- FORWARD PASS ---
        // 1. Create a one-hot encoded vector for the input character.
        const inputVector = createMatrix(1, vocabSize);
        inputVector.data[0][inputIndex] = 1;

        // 2. Calculate hidden layer activations.
        const hiddenRaw = add(dot(inputVector, model.hiddenLayer.weights), model.hiddenLayer.biases);
        const hiddenActivated = map(hiddenRaw, tanh);

        // 3. Calculate output layer logits (raw scores).
        const outputRaw = add(dot(hiddenActivated, model.outputLayer.weights), model.outputLayer.biases);
        const outputProbs = softmax(outputRaw); // Convert logits to probabilities.

        // 4. Calculate the loss (Cross-Entropy Loss).
        // This measures how "surprised" the model was by the correct answer.
        const loss = -Math.log(outputProbs.data[0][targetIndex] + 1e-9); // Add epsilon for stability.
        totalLoss += loss;

        // Store prediction results for logging.
        const predictedIndex = outputProbs.data[0].indexOf(Math.max(...outputProbs.data[0]));
        predictionResults.push({
            inputToken: model.vocab[inputIndex],
            targetToken: model.vocab[targetIndex],
            predictedToken: model.vocab[predictedIndex],
        });

        // --- BACKWARD PASS (Backpropagation) ---
        // This is where we calculate how much each weight and bias contributed to the error (loss).
        
        // 1. Gradient of the output probabilities. It's (probs - 1) for the target, and (probs) for others.
        // BUG FIX: Create a deep copy of outputProbs. Modifying it directly would corrupt the values sent to the UI.
        const dOutput = JSON.parse(JSON.stringify(outputProbs));
        dOutput.data[0][targetIndex] -= 1;

        // 2. Calculate gradients for the output layer.
        const dOutputWeights = dot(transpose(hiddenActivated), dOutput);
        const dOutputBiases = dOutput;
        
        // 3. Propagate the error back to the hidden layer.
        const dHiddenActivated = dot(dOutput, transpose(model.outputLayer.weights));
        const dHiddenRaw = multiply(dHiddenActivated, map(hiddenActivated, dtanh)); // Backprop through tanh.
        
        // 4. Calculate gradients for the hidden layer.
        const dHiddenWeights = dot(transpose(inputVector), dHiddenRaw);
        const dHiddenBiases = dHiddenRaw;
        
        // Accumulate gradients over the batch.
        hiddenGrad.weights = add(hiddenGrad.weights, dHiddenWeights);
        hiddenGrad.biases = add(hiddenGrad.biases, dHiddenBiases);
        outputGrad.weights = add(outputGrad.weights, dOutputWeights);
        outputGrad.biases = add(outputGrad.biases, dOutputBiases);

        // Save the last step's data for visualization.
        if (i === batchEnd - 1) {
            lastResultForVis = {
                inputToken: model.vocab[inputIndex],
                targetToken: model.vocab[targetIndex],
                predictedToken: model.vocab[predictedIndex],
                activations: { hidden: hiddenActivated, output: outputProbs, outputRaw: outputRaw },
            };
        }
    }
    
    const batchActualSize = batchEnd - step;
    if (batchActualSize === 0) {
      return { updatedModel: model, loss: 0, ...lastResultForVis, gradients: { hiddenLayer: hiddenGrad, outputLayer: outputGrad }, predictionResults: [] };
    }

    // --- WEIGHT UPDATE (Gradient Descent) ---
    // Update the model's parameters using the calculated gradients.
    // The learning rate controls the size of the update step.
    const clip = (m: Matrix) => map(m, v => Math.max(-5, Math.min(5, v))); // Gradient clipping to prevent explosion.
    const updatedModel = JSON.parse(JSON.stringify(model));
    const lr = -learningRate / batchActualSize; // Average gradient over batch.

    updatedModel.hiddenLayer.weights = add(model.hiddenLayer.weights, scale(clip(hiddenGrad.weights), lr));
    updatedModel.hiddenLayer.biases = add(model.hiddenLayer.biases, scale(clip(hiddenGrad.biases), lr));
    updatedModel.outputLayer.weights = add(model.outputLayer.weights, scale(clip(outputGrad.weights), lr));
    updatedModel.outputLayer.biases = add(model.outputLayer.biases, scale(clip(outputGrad.biases), lr));

    return {
        updatedModel,
        loss: totalLoss / batchActualSize,
        ...lastResultForVis,
        gradients: { hiddenLayer: hiddenGrad, outputLayer: outputGrad },
        predictionResults
    };
};

// A minimum word length to prevent the model from immediately generating a space.
const MIN_WORD_LENGTH = 2;

/**
 * Generates a word using a trained FFNN model.
 * @param model - The trained FFNN model.
 * @param seed - The starting token.
 * @param length - The maximum length of the word to generate.
 * @param temperature - Controls randomness. Higher values = more creative/random.
 * @returns The generated word.
 */
export const generateFFNN = (model: FFNNModel, seed: string, length: number, temperature: number = 0.7): string => {
    let result = '';
    let currentInput = seed;

    for (let i = 0; i < length; i++) {
        // Feed the last character of the current sequence into the model.
        const inputIndex = model.tokenToIndex[currentInput[currentInput.length - 1]];
        if (inputIndex === undefined) break;
        
        // Forward pass to get probabilities for the next character.
        const inputVector = createMatrix(1, model.vocab.length);
        inputVector.data[0][inputIndex] = 1;
        const hiddenActivated = map(add(dot(inputVector, model.hiddenLayer.weights), model.hiddenLayer.biases), tanh);
        const outputRaw = add(dot(hiddenActivated, model.outputLayer.weights), model.outputLayer.biases);
        
        // Apply temperature to the logits to control randomness.
        const scaledOutput = map(outputRaw, val => val / temperature);
        const outputProbs = softmax(scaledOutput);
        
        // Heuristic: Prevent generating a space too early to encourage word formation.
        if (i < MIN_WORD_LENGTH) {
            const spaceIndex = model.tokenToIndex[' '];
            if (spaceIndex !== undefined && outputProbs.data[0].length > spaceIndex) {
                const spaceProb = outputProbs.data[0][spaceIndex];
                if (spaceProb > 0 && spaceProb < 1) { // Avoid division by zero if space is the only option
                    outputProbs.data[0][spaceIndex] = 0;
                    // Renormalize other probabilities.
                    const remainingProbSum = 1 - spaceProb;
                    for (let k = 0; k < outputProbs.data[0].length; k++) {
                        if (k !== spaceIndex) {
                            outputProbs.data[0][k] /= remainingProbSum;
                        }
                    }
                }
            }
        }
        
        // Sample from the probability distribution to pick the next character.
        const rand = Math.random();
        let cumulativeProb = 0;
        let nextIndex = model.vocab.length - 1;
        for (let j = 0; j < outputProbs.data[0].length; j++) {
            cumulativeProb += outputProbs.data[0][j];
            if (rand < cumulativeProb) {
                nextIndex = j;
                break;
            }
        }
        
        const nextChar = model.vocab[nextIndex];
        if (nextChar === ' ') break; // Stop if a space is generated.
        result += nextChar;
        currentInput += nextChar;
    }
    return seed + result;
};


// --- RNN Implementation ---
export const initializeRNNModel = (vocab: string[], hiddenSize: number): RNNModel => {
    const vocabSize = vocab.length;
    const tokenToIndex = Object.fromEntries(vocab.map((char, i) => [char, i]));
    return {
        type: 'RNN',
        vocab,
        tokenToIndex,
        Wxh: createLayer(vocabSize, hiddenSize), // Input-to-Hidden weights
        Whh: createLayer(hiddenSize, hiddenSize), // Hidden-to-Hidden (recurrent) weights
        Why: createLayer(hiddenSize, vocabSize), // Hidden-to-Output weights
        h: createMatrix(1, hiddenSize), // Initial hidden state
    };
};

/**
 * Performs a single training step for an RNN over a sequence of characters.
 * This uses the Backpropagation Through Time (BPTT) algorithm.
 */
export const trainStepRNN = (
    model: RNNModel,
    encodedText: number[],
    step: number,
    sequenceLength: number,
    learningRate: number,
    dropoutRate: number = 0
): TrainStepResult => {
    const vocabSize = model.vocab.length;
    const hiddenSize = model.Wxh.weights.cols;
    const seqEnd = Math.min(step + sequenceLength, encodedText.length - 1);
    
    // Cache for storing activations at each time step, needed for backpropagation.
    const cache: any[] = [];
    const hiddenStates: Matrix[] = [model.h]; // Start with the model's current hidden state.
    let totalLoss = 0;
    const predictionResults: { inputToken: string; targetToken: string; predictedToken: string; }[] = [];

    // --- FORWARD PASS through the sequence ---
    for (let t = step; t < seqEnd; t++) {
        const inputVector = createMatrix(1, vocabSize);
        inputVector.data[0][encodedText[t]] = 1;

        const h_prev = hiddenStates[hiddenStates.length - 1];
        // Calculate the new hidden state: h_t = tanh(Wxh*x_t + Whh*h_{t-1} + biases)
        const h_next_raw = add(add(dot(inputVector, model.Wxh.weights), model.Wxh.biases), dot(h_prev, model.Whh.weights));
        const h_next = map(h_next_raw, tanh);

        // Apply dropout for regularization.
        let mask: Matrix | null = null;
        let h_next_dropped = h_next;
        if (dropoutRate > 0) {
            const scaleFactor = 1.0 / (1.0 - dropoutRate); // Inverted dropout scaling
            mask = createMatrix(1, hiddenSize, () => (Math.random() < dropoutRate ? 0 : scaleFactor));
            h_next_dropped = multiply(h_next, mask);
        }
        hiddenStates.push(h_next_dropped);

        // Calculate output probabilities: p = softmax(Why*h_t + biases)
        const outputRaw = add(dot(h_next_dropped, model.Why.weights), model.Why.biases);
        const prob = softmax(outputRaw);
        
        // Calculate loss for this time step.
        const targetIndex = encodedText[t + 1];
        totalLoss += -Math.log(prob.data[0][targetIndex] + 1e-9);
        
        const predictedIndex = prob.data[0].indexOf(Math.max(...prob.data[0]));
        predictionResults.push({
            inputToken: model.vocab[encodedText[t]],
            targetToken: model.vocab[targetIndex],
            predictedToken: model.vocab[predictedIndex],
        });

        // Store values needed for the backward pass.
        cache.push({ inputVector, h_prev, h_next, mask, prob, h_next_dropped });
    }
    
    // --- BACKWARD PASS (Backpropagation Through Time) ---
    // Initialize gradients to zero.
    let dWxh = createMatrix(vocabSize, hiddenSize);
    let dWhh = createMatrix(hiddenSize, hiddenSize);
    let dWhy = createMatrix(hiddenSize, vocabSize);
    let dbxh = createMatrix(1, hiddenSize);
    let dbhy = createMatrix(1, vocabSize);
    let dh_next_grad = createMatrix(1, hiddenSize); // Gradient from the *next* time step.

    // Iterate backwards through the sequence.
    for (let t = cache.length - 1; t >= 0; t--) {
        const { inputVector, h_prev, h_next, mask, prob, h_next_dropped } = cache[t];
        
        // BUG FIX: Deep copy `prob` to avoid modifying the cached value.
        const dy = JSON.parse(JSON.stringify(prob));
        dy.data[0][encodedText[step + t + 1]] -= 1; // Gradient for softmax

        // Calculate gradients for output layer.
        dWhy = add(dWhy, dot(transpose(h_next_dropped), dy));
        dbhy = add(dbhy, dy);

        // Backpropagate gradient to the hidden state.
        let dh = add(dot(dy, transpose(model.Why.weights)), dh_next_grad);
        
        if (mask) { // Backpropagate through dropout.
            dh = multiply(dh, mask);
        }

        // Backpropagate through the tanh activation function.
        let dh_raw = multiply(dh, map(h_next, dtanh));

        // Calculate gradients for hidden and input layers.
        dbxh = add(dbxh, dh_raw);
        dWxh = add(dWxh, dot(transpose(inputVector), dh_raw));
        dWhh = add(dWhh, dot(transpose(h_prev), dh_raw));
        // Pass the gradient to the previous time step.
        dh_next_grad = dot(dh_raw, transpose(model.Whh.weights));
    }
    
    // --- WEIGHT UPDATE ---
    const updatedModel = JSON.parse(JSON.stringify(model));
    const lr = -learningRate;
    const clip = (m: Matrix) => map(m, v => Math.max(-5, Math.min(5, v)));

    updatedModel.Wxh.weights = add(model.Wxh.weights, scale(clip(dWxh), lr));
    updatedModel.Whh.weights = add(model.Whh.weights, scale(clip(dWhh), lr));
    updatedModel.Why.weights = add(model.Why.weights, scale(clip(dWhy), lr));
    updatedModel.Wxh.biases = add(model.Wxh.biases, scale(clip(dbxh), lr));
    updatedModel.Why.biases = add(model.Why.biases, scale(clip(dbhy), lr));
    
    // Update the model's persistent hidden state for the next sequence.
    updatedModel.h = { ...hiddenStates[hiddenStates.length-1] };

    const lastProb = cache.length > 0 ? cache[cache.length-1].prob : null;
    const predictedIndex = lastProb ? lastProb.data[0].indexOf(Math.max(...lastProb.data[0])) : 0;
    
    return {
        updatedModel,
        loss: totalLoss / (seqEnd - step || 1),
        inputToken: model.vocab[encodedText[seqEnd-1]],
        targetToken: model.vocab[encodedText[seqEnd]],
        predictedToken: model.vocab[predictedIndex],
        predictionResults,
        activations: {
            hidden: hiddenStates[hiddenStates.length - 1],
            output: lastProb,
        },
    };
};

/**
 * Generates a word using a trained RNN model.
 */
export const generateRNN = (model: RNNModel, seed: string, length: number, temperature: number = 0.7): string => {
    let current_h = model.h; // Start with the model's last known hidden state.
    let result = '';
    let inputChar = seed;

    for (let i = 0; i < length; i++) {
        const inputIndex = model.tokenToIndex[inputChar];
        if (inputIndex === undefined) break;

        const inputVector = createMatrix(1, model.vocab.length);
        inputVector.data[0][inputIndex] = 1;
        
        // Forward pass for a single step.
        current_h = map(add(add(dot(inputVector, model.Wxh.weights), model.Wxh.biases), dot(current_h, model.Whh.weights)), tanh);
        const outputRaw = add(dot(current_h, model.Why.weights), model.Why.biases);

        // Sample the next character.
        const scaledOutput = map(outputRaw, val => val / temperature);
        const outputProbs = softmax(scaledOutput);
        
        if (i < MIN_WORD_LENGTH) {
            const spaceIndex = model.tokenToIndex[' '];
            if (spaceIndex !== undefined && outputProbs.data[0].length > spaceIndex) {
                const spaceProb = outputProbs.data[0][spaceIndex];
                if (spaceProb > 0 && spaceProb < 1) {
                    outputProbs.data[0][spaceIndex] = 0;
                    const remainingProbSum = 1 - spaceProb;
                    for (let k = 0; k < outputProbs.data[0].length; k++) {
                        if (k !== spaceIndex) {
                           outputProbs.data[0][k] /= remainingProbSum;
                        }
                    }
                }
            }
        }
        
        const rand = Math.random();
        let cumulativeProb = 0;
        let nextIndex = model.vocab.length - 1;
        for (let j = 0; j < outputProbs.data[0].length; j++) {
            cumulativeProb += outputProbs.data[0][j];
            if (rand < cumulativeProb) {
                nextIndex = j;
                break;
            }
        }
        
        const nextChar = model.vocab[nextIndex];
        if (nextChar === ' ') break;
        result += nextChar;
        inputChar = nextChar; // The output becomes the next input.
    }

    return seed + result;
};


// --- GRU Implementation ---
export const initializeGRUModel = (vocab: string[], hiddenSize: number): GRULanguageModel => {
    const vocabSize = vocab.length;
    const tokenToIndex = Object.fromEntries(vocab.map((char, i) => [char, i]));
    return {
        type: 'GRU',
        vocab,
        tokenToIndex,
        Wz: createLayer(vocabSize, hiddenSize), // Update gate
        Uz: createLayer(hiddenSize, hiddenSize),
        Wr: createLayer(vocabSize, hiddenSize), // Reset gate
        Ur: createLayer(hiddenSize, hiddenSize),
        Wh: createLayer(vocabSize, hiddenSize), // Candidate state
        Uh: createLayer(hiddenSize, hiddenSize),
        Why: createLayer(hiddenSize, vocabSize), // Output layer
        h: createMatrix(1, hiddenSize), // Initial hidden state
    };
};

/**
 * Performs a single training step for a GRU model.
 */
export const trainStepGRU = (
    model: GRULanguageModel,
    encodedText: number[],
    step: number,
    sequenceLength: number,
    learningRate: number,
    dropoutRate: number = 0
): TrainStepResult => {
    const vocabSize = model.vocab.length;
    const hiddenSize = model.Why.weights.rows;
    const seqEnd = Math.min(step + sequenceLength, encodedText.length - 1);
    
    const cache: any[] = [];
    const hiddenStates: Matrix[] = [model.h];
    let totalLoss = 0;
    const predictionResults: { inputToken: string; targetToken: string; predictedToken: string; }[] = [];

    // --- FORWARD PASS ---
    for (let t = step; t < seqEnd; t++) {
        const inputVector = createMatrix(1, vocabSize);
        inputVector.data[0][encodedText[t]] = 1;

        const h_prev = hiddenStates[hiddenStates.length - 1];

        // GRU gate calculations
        // z_t (update gate): decides how much of the past information to keep.
        const z_t = map(add(add(dot(inputVector, model.Wz.weights), model.Wz.biases), dot(h_prev, model.Uz.weights)), sigmoid);
        // r_t (reset gate): decides how much of the past information to forget.
        const r_t = map(add(add(dot(inputVector, model.Wr.weights), model.Wr.biases), dot(h_prev, model.Ur.weights)), sigmoid);
        // h_hat_t (candidate hidden state): a new hidden state proposed based on the input and *reset* previous state.
        const h_hat_t = map(add(add(dot(inputVector, model.Wh.weights), model.Wh.biases), dot(multiply(r_t, h_prev), model.Uh.weights)), tanh);
        // h_t (final hidden state): a combination of the previous state and the candidate state, controlled by the update gate.
        const h_t = add(multiply(subtract(ones(1, hiddenSize), z_t), h_prev), multiply(z_t, h_hat_t));
        
        let mask: Matrix | null = null;
        let h_t_dropped = h_t;
        if (dropoutRate > 0) {
            const scaleFactor = 1.0 / (1.0 - dropoutRate);
            mask = createMatrix(1, hiddenSize, () => (Math.random() < dropoutRate ? 0 : scaleFactor));
            h_t_dropped = multiply(h_t, mask);
        }
        hiddenStates.push(h_t_dropped);

        // Output calculation
        const outputRaw = add(dot(h_t_dropped, model.Why.weights), model.Why.biases);
        const prob = softmax(outputRaw);
        
        const targetIndex = encodedText[t + 1];
        totalLoss += -Math.log(prob.data[0][targetIndex] + 1e-9);

        const predictedIndex = prob.data[0].indexOf(Math.max(...prob.data[0]));
        predictionResults.push({
            inputToken: model.vocab[encodedText[t]],
            targetToken: model.vocab[targetIndex],
            predictedToken: model.vocab[predictedIndex],
        });

        cache.push({ inputVector, h_prev, z_t, r_t, h_hat_t, h_t, mask, prob, h_t_dropped });
    }
    
    // --- BACKWARD PASS ---
    // Initialize gradients
    const grads = {
        Wz: createMatrix(vocabSize, hiddenSize), Uz: createMatrix(hiddenSize, hiddenSize), bz: createMatrix(1, hiddenSize),
        Wr: createMatrix(vocabSize, hiddenSize), Ur: createMatrix(hiddenSize, hiddenSize), br: createMatrix(1, hiddenSize),
        Wh: createMatrix(vocabSize, hiddenSize), Uh: createMatrix(hiddenSize, hiddenSize), bh: createMatrix(1, hiddenSize),
        Why: createMatrix(hiddenSize, vocabSize), by: createMatrix(1, vocabSize),
    };
    let dh_next_grad = createMatrix(1, hiddenSize);

    for (let t = cache.length - 1; t >= 0; t--) {
        const { inputVector, h_prev, z_t, r_t, h_hat_t, h_t, mask, prob } = cache[t];
        
        // BUG FIX: Deep copy `prob` to avoid modifying the cached value.
        const dy = JSON.parse(JSON.stringify(prob));
        dy.data[0][encodedText[step + t + 1]] -= 1;

        // Gradients for output layer
        grads.Why = add(grads.Why, dot(transpose(hiddenStates[t + 1]), dy));
        grads.by = add(grads.by, dy);

        // Backpropagate to hidden state
        let dh = add(dot(dy, transpose(model.Why.weights)), dh_next_grad);
        if (mask) { dh = multiply(dh, mask); }
        
        // Backpropagate through the final hidden state equation
        const dh_prev_from_h = multiply(dh, subtract(ones(1, hiddenSize), z_t));
        const dz = multiply(dh, subtract(h_hat_t, h_prev));
        const dh_hat = multiply(dh, z_t);
        
        // Backpropagate through candidate state
        const dh_hat_raw = multiply(dh_hat, map(h_hat_t, dtanh));
        grads.Wh = add(grads.Wh, dot(transpose(inputVector), dh_hat_raw));
        grads.bh = add(grads.bh, dh_hat_raw);
        grads.Uh = add(grads.Uh, dot(transpose(multiply(r_t, h_prev)), dh_hat_raw));
        
        // Backpropagate through reset gate
        const dr_h_prev = dot(dh_hat_raw, transpose(model.Uh.weights));
        const dr = multiply(dr_h_prev, h_prev);
        const dr_raw = multiply(dr, map(r_t, dsigmoid));
        grads.Wr = add(grads.Wr, dot(transpose(inputVector), dr_raw));
        grads.br = add(grads.br, dr_raw);
        grads.Ur = add(grads.Ur, dot(transpose(h_prev), dr_raw));
        
        // Backpropagate through update gate
        const dz_raw = multiply(dz, map(z_t, dsigmoid));
        grads.Wz = add(grads.Wz, dot(transpose(inputVector), dz_raw));
        grads.bz = add(grads.bz, dz_raw);
        grads.Uz = add(grads.Uz, dot(transpose(h_prev), dz_raw));
        
        // Accumulate gradient for the *next* dh_next_grad
        const dh_prev_from_r = multiply(dr_h_prev, r_t);
        const dh_prev_from_z = dot(dz_raw, transpose(model.Uz.weights));
        const dh_prev_from_r_gate = dot(dr_raw, transpose(model.Ur.weights));
        dh_next_grad = add(add(add(dh_prev_from_h, dh_prev_from_r), dh_prev_from_z), dh_prev_from_r_gate);
    }
    
    // --- WEIGHT UPDATE ---
    const updatedModel = JSON.parse(JSON.stringify(model));
    const lr = -learningRate;
    const clip = (m: Matrix) => map(m, v => Math.max(-5, Math.min(5, v)));

    updatedModel.Wz.weights = add(model.Wz.weights, scale(clip(grads.Wz), lr));
    updatedModel.Uz.weights = add(model.Uz.weights, scale(clip(grads.Uz), lr));
    updatedModel.Wz.biases = add(model.Wz.biases, scale(clip(grads.bz), lr));
    updatedModel.Wr.weights = add(model.Wr.weights, scale(clip(grads.Wr), lr));
    updatedModel.Ur.weights = add(model.Ur.weights, scale(clip(grads.Ur), lr));
    updatedModel.Wr.biases = add(model.Wr.biases, scale(clip(grads.br), lr));
    updatedModel.Wh.weights = add(model.Wh.weights, scale(clip(grads.Wh), lr));
    updatedModel.Uh.weights = add(model.Uh.weights, scale(clip(grads.Uh), lr));
    updatedModel.Wh.biases = add(model.Wh.biases, scale(clip(grads.bh), lr));
    updatedModel.Why.weights = add(model.Why.weights, scale(clip(grads.Why), lr));
    updatedModel.Why.biases = add(model.Why.biases, scale(clip(grads.by), lr));
    
    updatedModel.h = { ...hiddenStates[hiddenStates.length-1] };

    const lastCacheEntry = cache.length > 0 ? cache[cache.length - 1] : null;
    const lastProb = lastCacheEntry ? lastCacheEntry.prob : null;
    const predictedIndex = lastProb ? lastProb.data[0].indexOf(Math.max(...lastProb.data[0])) : 0;
    
    return {
        updatedModel,
        loss: totalLoss / (seqEnd - step || 1),
        inputToken: model.vocab[encodedText[seqEnd-1]],
        targetToken: model.vocab[encodedText[seqEnd]],
        predictedToken: model.vocab[predictedIndex],
        predictionResults,
        activations: {
            hidden: hiddenStates[hiddenStates.length - 1],
            output: lastProb,
        },
        gateActivations: {
            z: lastCacheEntry ? lastCacheEntry.z_t : createMatrix(1, hiddenSize),
            r: lastCacheEntry ? lastCacheEntry.r_t : createMatrix(1, hiddenSize),
        }
    };
};

/**
 * Generates a word using a trained GRU model.
 */
export const generateGRU = (model: GRULanguageModel, seed: string, length: number, temperature: number = 0.7): string => {
    let h_prev = model.h;
    let result = '';
    let inputChar = seed;

    for (let i = 0; i < length; i++) {
        const inputIndex = model.tokenToIndex[inputChar];
        if (inputIndex === undefined) break;
        
        const inputVector = createMatrix(1, model.vocab.length);
        inputVector.data[0][inputIndex] = 1;
        
        // Single forward pass step for generation.
        const z_t = map(add(add(dot(inputVector, model.Wz.weights), model.Wz.biases), dot(h_prev, model.Uz.weights)), sigmoid);
        const r_t = map(add(add(dot(inputVector, model.Wr.weights), model.Wr.biases), dot(h_prev, model.Ur.weights)), sigmoid);
        const h_hat_t = map(add(add(dot(inputVector, model.Wh.weights), model.Wh.biases), dot(multiply(r_t, h_prev), model.Uh.weights)), tanh);
        const h_t = add(multiply(subtract(ones(1, h_prev.cols), z_t), h_prev), multiply(z_t, h_hat_t));
        
        const outputRaw = add(dot(h_t, model.Why.weights), model.Why.biases);

        // Sample next character.
        const scaledOutput = map(outputRaw, val => val / temperature);
        const outputProbs = softmax(scaledOutput);
        
        if (i < MIN_WORD_LENGTH) {
            const spaceIndex = model.tokenToIndex[' '];
            if (spaceIndex !== undefined && outputProbs.data[0].length > spaceIndex) {
                const spaceProb = outputProbs.data[0][spaceIndex];
                if (spaceProb > 0 && spaceProb < 1) {
                    outputProbs.data[0][spaceIndex] = 0;
                    const remainingProbSum = 1 - spaceProb;
                    for (let k = 0; k < outputProbs.data[0].length; k++) {
                       if (k !== spaceIndex) {
                           outputProbs.data[0][k] /= remainingProbSum;
                       }
                    }
                }
            }
        }
        
        const rand = Math.random();
        let cumulativeProb = 0;
        let nextIndex = model.vocab.length - 1;
        for (let j = 0; j < outputProbs.data[0].length; j++) {
            cumulativeProb += outputProbs.data[0][j];
            if (rand < cumulativeProb) {
                nextIndex = j;
                break;
            }
        }
        
        const nextChar = model.vocab[nextIndex];
        if (nextChar === ' ') break;
        result += nextChar;
        inputChar = nextChar;
        h_prev = h_t;
    }

    return seed + result;
};

// --- LSTM Implementation ---
export const initializeLSTMModel = (vocab: string[], hiddenSize: number): LSTMLanguageModel => {
    const vocabSize = vocab.length;
    const tokenToIndex = Object.fromEntries(vocab.map((char, i) => [char, i]));
    return {
        type: 'LSTM',
        vocab,
        tokenToIndex,
        Wf: createLayer(vocabSize, hiddenSize), Uf: createLayer(hiddenSize, hiddenSize), // Forget gate
        Wi: createLayer(vocabSize, hiddenSize), Ui: createLayer(hiddenSize, hiddenSize), // Input gate
        Wo: createLayer(vocabSize, hiddenSize), Uo: createLayer(hiddenSize, hiddenSize), // Output gate
        Wc: createLayer(vocabSize, hiddenSize), Uc: createLayer(hiddenSize, hiddenSize), // Cell state candidate
        Why: createLayer(hiddenSize, vocabSize), // Output layer
        h: createMatrix(1, hiddenSize), // Initial hidden state
        c: createMatrix(1, hiddenSize), // Initial cell state
    };
};

/**
 * Performs a single training step for an LSTM model.
 */
export const trainStepLSTM = (
    model: LSTMLanguageModel,
    encodedText: number[],
    step: number,
    sequenceLength: number,
    learningRate: number,
    dropoutRate: number = 0
): TrainStepResult => {
    const vocabSize = model.vocab.length;
    const hiddenSize = model.Why.weights.rows;
    const seqEnd = Math.min(step + sequenceLength, encodedText.length - 1);

    const cache: any[] = [];
    const hiddenStates: Matrix[] = [model.h];
    const cellStates: Matrix[] = [model.c];
    let totalLoss = 0;
    const predictionResults: { inputToken: string; targetToken: string; predictedToken: string; }[] = [];

    // --- FORWARD PASS ---
    for (let t = step; t < seqEnd; t++) {
        const inputVector = createMatrix(1, vocabSize);
        inputVector.data[0][encodedText[t]] = 1;
        const h_prev = hiddenStates[hiddenStates.length - 1];
        const c_prev = cellStates[cellStates.length - 1];

        // LSTM gate and state calculations
        // f_t (forget gate): decides what to throw away from the old cell state.
        const f_t = map(add(add(dot(inputVector, model.Wf.weights), model.Wf.biases), dot(h_prev, model.Uf.weights)), sigmoid);
        // i_t (input gate): decides which new values to update in the cell state.
        const i_t = map(add(add(dot(inputVector, model.Wi.weights), model.Wi.biases), dot(h_prev, model.Ui.weights)), sigmoid);
        // o_t (output gate): decides what part of the cell state to output as the new hidden state.
        const o_t = map(add(add(dot(inputVector, model.Wo.weights), model.Wo.biases), dot(h_prev, model.Uo.weights)), sigmoid);
        // c_hat_t (candidate cell state): a vector of new candidate values to be added to the cell state.
        const c_hat_t = map(add(add(dot(inputVector, model.Wc.weights), model.Wc.biases), dot(h_prev, model.Uc.weights)), tanh);
        
        // c_t (new cell state): combination of forgetting old parts and adding new parts.
        const c_t = add(multiply(f_t, c_prev), multiply(i_t, c_hat_t));
        // h_t (new hidden state): filtered version of the new cell state.
        const h_t = multiply(o_t, map(c_t, tanh));

        let mask: Matrix | null = null;
        let h_t_dropped = h_t;
        if (dropoutRate > 0) {
            const scaleFactor = 1.0 / (1.0 - dropoutRate);
            mask = createMatrix(1, hiddenSize, () => (Math.random() < dropoutRate ? 0 : scaleFactor));
            h_t_dropped = multiply(h_t, mask);
        }
        hiddenStates.push(h_t_dropped);
        cellStates.push(c_t);

        // Output calculation
        const outputRaw = add(dot(h_t_dropped, model.Why.weights), model.Why.biases);
        const prob = softmax(outputRaw);
        const targetIndex = encodedText[t + 1];
        totalLoss += -Math.log(prob.data[0][targetIndex] + 1e-9);

        const predictedIndex = prob.data[0].indexOf(Math.max(...prob.data[0]));
        predictionResults.push({
            inputToken: model.vocab[encodedText[t]],
            targetToken: model.vocab[targetIndex],
            predictedToken: model.vocab[predictedIndex],
        });

        cache.push({ inputVector, h_prev, c_prev, f_t, i_t, o_t, c_hat_t, c_t, h_t, mask, prob });
    }

    // --- BACKWARD PASS ---
    // Initialize gradients
    const grads = {
        Wf: createMatrix(vocabSize, hiddenSize), Uf: createMatrix(hiddenSize, hiddenSize), bf: createMatrix(1, hiddenSize),
        Wi: createMatrix(vocabSize, hiddenSize), Ui: createMatrix(hiddenSize, hiddenSize), bi: createMatrix(1, hiddenSize),
        Wo: createMatrix(vocabSize, hiddenSize), Uo: createMatrix(hiddenSize, hiddenSize), bo: createMatrix(1, hiddenSize),
        Wc: createMatrix(vocabSize, hiddenSize), Uc: createMatrix(hiddenSize, hiddenSize), bc: createMatrix(1, hiddenSize),
        Why: createMatrix(hiddenSize, vocabSize), by: createMatrix(1, vocabSize),
    };
    let dh_next_grad = createMatrix(1, hiddenSize);
    let dc_next_grad = createMatrix(1, hiddenSize);

    for (let t = cache.length - 1; t >= 0; t--) {
        const { inputVector, h_prev, c_prev, f_t, i_t, o_t, c_hat_t, c_t, h_t, mask, prob } = cache[t];
        
        // BUG FIX: Deep copy `prob` to avoid modifying the cached value.
        const dy = JSON.parse(JSON.stringify(prob));
        dy.data[0][encodedText[step + t + 1]] -= 1;
        
        // Gradients for output layer
        grads.Why = add(grads.Why, dot(transpose(hiddenStates[t + 1]), dy));
        grads.by = add(grads.by, dy);

        // Backpropagate to hidden state
        let dh = add(dot(dy, transpose(model.Why.weights)), dh_next_grad);
        if (mask) { dh = multiply(dh, mask); }

        // Backpropagate through output gate
        const do_raw = multiply(dh, map(c_t, tanh));
        const do_t = multiply(do_raw, map(o_t, dsigmoid));
        grads.Wo = add(grads.Wo, dot(transpose(inputVector), do_t));
        grads.Uo = add(grads.Uo, dot(transpose(h_prev), do_t));
        grads.bo = add(grads.bo, do_t);

        // Backpropagate to cell state
        const dc_t = add(dc_next_grad, multiply(dh, multiply(o_t, map(map(c_t, tanh), dtanh))));

        // Backpropagate through candidate cell state
        const dc_hat_t = multiply(dc_t, i_t);
        const dc_hat_raw = multiply(dc_hat_t, map(c_hat_t, dtanh));
        grads.Wc = add(grads.Wc, dot(transpose(inputVector), dc_hat_raw));
        grads.Uc = add(grads.Uc, dot(transpose(h_prev), dc_hat_raw));
        grads.bc = add(grads.bc, dc_hat_raw);
        
        // Backpropagate through input gate
        const di_t = multiply(dc_t, c_hat_t);
        const di_raw = multiply(di_t, map(i_t, dsigmoid));
        grads.Wi = add(grads.Wi, dot(transpose(inputVector), di_raw));
        grads.Ui = add(grads.Ui, dot(transpose(h_prev), di_raw));
        grads.bi = add(grads.bi, di_raw);

        // Backpropagate through forget gate
        const df_t = multiply(dc_t, c_prev);
        const df_raw = multiply(df_t, map(f_t, dsigmoid));
        grads.Wf = add(grads.Wf, dot(transpose(inputVector), df_raw));
        grads.Uf = add(grads.Uf, dot(transpose(h_prev), df_raw));
        grads.bf = add(grads.bf, df_raw);

        // Pass gradients to the next (previous in time) step
        dc_next_grad = multiply(dc_t, f_t);
        dh_next_grad = add(add(add(dot(df_raw, transpose(model.Uf.weights)), dot(di_raw, transpose(model.Ui.weights))), dot(do_raw, transpose(model.Uo.weights))), dot(dc_hat_raw, transpose(model.Uc.weights)));
    }
    
    // --- WEIGHT UPDATE ---
    const updatedModel = JSON.parse(JSON.stringify(model));
    const lr = -learningRate;
    const clip = (m: Matrix) => map(m, v => Math.max(-5, Math.min(5, v)));
    
    // Helper to apply gradients to a gate's parameters (e.g., Wf, Uf, bf)
    const applyGrads = (gateChar: 'f' | 'i' | 'o' | 'c') => {
      const W_key = `W${gateChar}` as keyof LSTMLanguageModel;
      const U_key = `U${gateChar}` as keyof LSTMLanguageModel;
      (updatedModel as any)[W_key].weights = add((model as any)[W_key].weights, scale(clip(grads[W_key]), lr));
      (updatedModel as any)[U_key].weights = add((model as any)[U_key].weights, scale(clip(grads[U_key]), lr));
      (updatedModel as any)[W_key].biases = add((model as any)[W_key].biases, scale(clip(grads[`b${gateChar}` as keyof typeof grads]), lr));
    };

    applyGrads('f'); applyGrads('i'); applyGrads('o'); applyGrads('c');
    updatedModel.Why.weights = add(model.Why.weights, scale(clip(grads.Why), lr));
    updatedModel.Why.biases = add(model.Why.biases, scale(clip(grads.by), lr));
    
    updatedModel.h = { ...hiddenStates[hiddenStates.length-1] };
    updatedModel.c = { ...cellStates[cellStates.length-1] };

    const lastCacheEntry = cache.length > 0 ? cache[cache.length - 1] : null;
    const lastProb = lastCacheEntry ? lastCacheEntry.prob : null;
    const predictedIndex = lastProb ? lastProb.data[0].indexOf(Math.max(...lastProb.data[0])) : 0;
    
    return {
        updatedModel,
        loss: totalLoss / (seqEnd - step || 1),
        inputToken: model.vocab[encodedText[seqEnd-1]],
        targetToken: model.vocab[encodedText[seqEnd]],
        predictedToken: model.vocab[predictedIndex],
        predictionResults,
        activations: {
            hidden: hiddenStates[hiddenStates.length - 1],
            output: lastProb,
        },
        gateActivations: {
            f: lastCacheEntry ? lastCacheEntry.f_t : createMatrix(1, hiddenSize),
            i: lastCacheEntry ? lastCacheEntry.i_t : createMatrix(1, hiddenSize),
            o: lastCacheEntry ? lastCacheEntry.o_t : createMatrix(1, hiddenSize),
        }
    };
};

/**
 * Generates a word using a trained LSTM model.
 */
export const generateLSTM = (model: LSTMLanguageModel, seed: string, length: number, temperature: number = 0.7): string => {
    let h_prev = model.h;
    let c_prev = model.c;
    let result = '';
    let inputChar = seed;

    for (let i = 0; i < length; i++) {
        const inputIndex = model.tokenToIndex[inputChar];
        if (inputIndex === undefined) break;
        
        const inputVector = createMatrix(1, model.vocab.length);
        inputVector.data[0][inputIndex] = 1;
        
        // Single forward pass step for generation.
        const f_t = map(add(add(dot(inputVector, model.Wf.weights), model.Wf.biases), dot(h_prev, model.Uf.weights)), sigmoid);
        const i_t = map(add(add(dot(inputVector, model.Wi.weights), model.Wi.biases), dot(h_prev, model.Ui.weights)), sigmoid);
        const o_t = map(add(add(dot(inputVector, model.Wo.weights), model.Wo.biases), dot(h_prev, model.Uo.weights)), sigmoid);
        const c_hat_t = map(add(add(dot(inputVector, model.Wc.weights), model.Wc.biases), dot(h_prev, model.Uc.weights)), tanh);
        
        const c_t = add(multiply(f_t, c_prev), multiply(i_t, c_hat_t));
        const h_t = multiply(o_t, map(c_t, tanh));
        
        const outputRaw = add(dot(h_t, model.Why.weights), model.Why.biases);
        const scaledOutput = map(outputRaw, val => val / temperature);
        const outputProbs = softmax(scaledOutput);
        
        if (i < MIN_WORD_LENGTH) {
            const spaceIndex = model.tokenToIndex[' '];
            if (spaceIndex !== undefined && outputProbs.data[0].length > spaceIndex) {
                const spaceProb = outputProbs.data[0][spaceIndex];
                if (spaceProb > 0 && spaceProb < 1) {
                    outputProbs.data[0][spaceIndex] = 0;
                    const remainingProbSum = 1 - spaceProb;
                    for (let k = 0; k < outputProbs.data[0].length; k++) {
                        if (k !== spaceIndex) {
                            outputProbs.data[0][k] /= remainingProbSum;
                        }
                    }
                }
            }
        }
        
        // Sample the next character.
        const rand = Math.random();
        let cumulativeProb = 0;
        let nextIndex = model.vocab.length - 1;
        for (let j = 0; j < outputProbs.data[0].length; j++) {
            cumulativeProb += outputProbs.data[0][j];
            if (rand < cumulativeProb) {
                nextIndex = j;
                break;
            }
        }
        
        const nextChar = model.vocab[nextIndex];
        if (nextChar === ' ') break;
        result += nextChar;
        inputChar = nextChar;
        h_prev = h_t;
        c_prev = c_t;
    }

    return seed + result;
};
