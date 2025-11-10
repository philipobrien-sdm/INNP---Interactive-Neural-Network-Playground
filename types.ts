/**
 * @file types.ts
 * @description This file defines the core data structures and types used throughout the application.
 * By centralizing these definitions, we ensure type safety and a clear understanding of the
 * data being passed between components and services.
 */

// FIX: Removed an invalid file marker from the beginning of the file which was causing a syntax error.


// FIX: Removed faulty import and defined BpeMerges locally.
/**
 * Represents the learned merges for Byte-Pair Encoding (BPE).
 * The key is a string like "101,103" representing a pair of token IDs,
 * and the value is the new merged token ID.
 */
export type BpeMerges = Map<string, number>;

/**
 * Represents a 2D matrix, the fundamental data structure for neural network parameters.
 */
export interface Matrix {
  rows: number;      // Number of rows in the matrix.
  cols: number;      // Number of columns in the matrix.
  data: number[][]; // The actual numerical data, stored as a 2D array.
}

/**
 * Represents a single layer in a neural network, consisting of weights and biases.
 */
export interface Layer {
  weights: Matrix; // Matrix of weights connecting the previous layer to this one.
  biases: Matrix;  // Matrix of biases added to each neuron in this layer.
}

/**
 * Defines the structure for a Feed-Forward Neural Network (FFNN) model.
 */
export interface FFNNModel {
  type: 'FFNN'; // A discriminator to identify the model type.
  vocab: string[]; // An array of all unique tokens the model knows.
  tokenToIndex: { [key: string]: number }; // A mapping from a token string to its integer index.
  hiddenLayer: Layer; // The single hidden layer of the network.
  outputLayer: Layer; // The final output layer.
}

/**
 * Defines the structure for a Recurrent Neural Network (RNN) model.
 */
export interface RNNModel {
  type: 'RNN';
  vocab: string[];
  tokenToIndex: { [key: string]: number };
  Wxh: Layer; // Weights from input (x) to hidden (h).
  Whh: Layer; // Weights from hidden (h) to hidden (h) - the recurrent connection.
  Why: Layer; // Weights from hidden (h) to output (y).
  h: Matrix;   // The hidden state, which acts as the model's memory.
}

/**
 * Defines the structure for a Gated Recurrent Unit (GRU) model.
 */
export interface GRULanguageModel {
    type: 'GRU';
    vocab: string[];
    tokenToIndex: { [key: string]: number };
    // Update gate parameters (controls how much of the past to keep).
    Wz: Layer;
    Uz: Layer;
    // Reset gate parameters (controls how much to forget).
    Wr: Layer;
    Ur: Layer;
    // Candidate hidden state parameters (proposes a new hidden state).
    Wh: Layer;
    Uh: Layer;
    // Output layer parameters.
    Why: Layer;
    // The hidden state memory.
    h: Matrix;
}

/**
 * Defines the structure for a Long Short-Term Memory (LSTM) model.
 */
export interface LSTMLanguageModel {
    type: 'LSTM';
    vocab: string[];
    tokenToIndex: { [key: string]: number };
    // Forget gate parameters (decides what to discard from the cell state).
    Wf: Layer; Uf: Layer;
    // Input gate parameters (decides what new information to store).
    Wi: Layer; Ui: Layer;
    // Output gate parameters (decides what to output based on the cell state).
    Wo: Layer; Uo: Layer;
    // Cell gate parameters (creates candidate values for the cell state).
    Wc: Layer; Uc: Layer;
    // Output layer parameters.
    Why: Layer;
    // The hidden state (short-term memory) and cell state (long-term memory).
    h: Matrix;
    c: Matrix;
}


/**
 * A union type representing any of the possible language models in the app.
 * This allows for polymorphic handling of different model architectures.
 */
export type LanguageModel = FFNNModel | RNNModel | GRULanguageModel | LSTMLanguageModel;

/**
 * Represents the complete result of a single training step.
 * This object is used to update the model, UI visualizations, and logs.
 */
export interface TrainStepResult {
    updatedModel: LanguageModel; // The model with its weights adjusted after this step.
    loss: number; // The calculated error for this step (lower is better).
    inputToken: string; // The input token for this step.
    targetToken: string; // The correct "next" token the model was supposed to predict.
    predictedToken: string; // The token the model actually predicted.
    // A log of all predictions made within the batch/sequence for this step.
    predictionResults: {
        inputToken: string;
        targetToken: string;
        predictedToken: string;
    }[];
    // The activation values of the neurons at different stages, used for visualization.
    activations: {
        hidden: Matrix; // Activations of the hidden layer neurons.
        output: Matrix; // Probabilities from the output layer (after softmax).
        outputRaw?: Matrix; // Raw logits from the output layer (before softmax), for demo purposes.
    };
    // The calculated gradients for each layer, used for the interactive demo.
    gradients?: any;
    // For GRU/LSTM, the activation values of their specific gates.
    gateActivations?: {
        z?: Matrix; // GRU update gate
        r?: Matrix; // GRU reset gate
        f?: Matrix; // LSTM forget gate
        i?: Matrix; // LSTM input gate
        o?: Matrix; // LSTM output gate
    };
}

/**
 * Represents a snapshot of generated words at a specific point in training.
 */
export interface GenerationHistoryItem {
  epoch: number;   // The epoch number when this generation occurred.
  words: string[]; // The list of words generated by the model.
}

/**
 * Represents a BPE tokenizer's state.
 */
export interface Tokenizer {
    merges: BpeMerges; // The learned merge rules.
    vocab: { [key: number]: string }; // Mapping from token ID to token string.
}
