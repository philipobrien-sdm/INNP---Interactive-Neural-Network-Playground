
# Interactive Neural Network Playground

**Train character-level language models from scratch, right in your browser. An educational tool to demystify how AI learns language.**

---

### 🚀 [View the Live Demo on Google AI Studio](https://aistudio.google.com/app/project/shared-L1zhcWpL-D9V) 🚀

*(Note: The link above will take you to Google AI Studio where you can run the application directly in your browser.)*

![Interactive Neural Network Playground Demo](https://storage.googleapis.com/aistudio-project-assets/github_readme_assets/nn-playground-demo.gif)

## About The Project

This application is an interactive, educational tool designed to demystify the basics of how different neural network architectures learn to generate text. You are training a character-level language model from scratch, watching it learn the fundamental patterns of word structure (phonotactics) from a block of text, much like a child learns which sounds can go together.

The project visualizes every part of the process, from the internal "thoughts" of the network (neuron activations) to its improving performance over time (loss charts). It progresses through four key architectures, each building on the concepts of the last:

1.  **FFNN (Feed-Forward Neural Network):** A memoryless network that learns simple, direct relationships.
2.  **RNN (Recurrent Neural Network):** Introduces a simple memory loop, allowing it to learn from sequences.
3.  **GRU (Gated Recurrent Unit):** An advanced RNN with "gates" to intelligently manage its memory.
4.  **LSTM (Long Short-Term Memory):** The most powerful architecture here, with a sophisticated system for tracking both long and short-term memory.

The core neural network logic, including all the matrix math for forward and backward passes, is implemented **from scratch** in `services/languageModel.ts` to make the learning process as transparent as possible.

## Key Features

-   **Four Model Architectures:** Train and compare FFNN, RNN, GRU, and LSTM models side-by-side.
-   **Interactive Visualizations:**
    -   **Live Architecture View:** See neuron activations and weight matrices update in real-time as heatmaps.
    -   **Step-by-Step Demo:** A detailed, slow-motion view of the FFNN's internal math for a single training step.
    -   **Loss & Accuracy Charts:** Track your model's performance with a loss histogram and a prediction success-rate heatmap.
-   **Hyperparameter Tuning:** Interactively adjust the Learning Rate, Hidden Size, Sequence Length, and more to see their immediate impact on training.
-   **Advanced Tokenization:** Switch between Character, Byte-Pair Encoding (BPE), and a custom phonotactics-based tokenizer.
-   **Generation & Coaching:** Generate words from your trained model and provide feedback. The "Auto Coach" feature automates this process, using a sophisticated phonotactic validator (`services/wordValidator.ts`) to find and reinforce "good" words.
-   **Cyclical Training:** Automate a full cycle of training and auto-coaching to create a powerful feedback loop for model improvement.
-   **Save & Load Models:** Save your trained model's state to a file and load it back later to continue your work.

## How to Run on Google AI Studio

Viewing and running this project is simple with Google AI Studio.

1.  **Click the Link:** Click this link to open the project in AI Studio:
    > ### [https://aistudio.google.com/app/project/shared-L1zhcWpL-D9V](https://aistudio.google.com/app/project/shared-L1zhcWpL-D9V)

2.  **Open the Project:** AI Studio will load the project's code and environment.

3.  **Run the App:** The application will automatically build and launch in the preview panel on the right-hand side of the screen.

4.  **Interact:** You can now interact with the live application!
    -   Start with the **Interactive Demo** tab for a guided tour of the fundamentals.
    -   Move to the **Simple (FFNN)** tab and click "Start Training" to see the full process in action.
    -   Explore the other tabs to see how memory (RNN, GRU, LSTM) improves the model's ability to learn.

## Technical Deep Dive

For those interested in the code, here are the key files to look at:

-   `components/Playground.tsx`: The main stateful React component that orchestrates the entire training process, UI, and visualizations for the FFNN, RNN, GRU, and LSTM tabs.
-   `services/languageModel.ts`: **The heart of the project.** This file contains the from-scratch TypeScript implementation of all four neural network architectures, including matrix math, activation functions, the forward pass (prediction), and the backward pass (learning via backpropagation).
-   `components/InteractiveDemo.tsx`: The self-contained component for the detailed, step-by-step FFNN visualization.
-   `services/wordValidator.ts`: Contains the rule-based engine based on English phonotactics that determines if a generated word is "good" for the Auto Coach feature.
-   `services/bpe.ts` & `services/customTokenizer.ts`: The logic for the BPE and Custom tokenization methods.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
