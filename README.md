# Interactive Neural Network Playground

**Train character-level language models from scratch, right in your browser. An educational tool to demystify how AI learns language.**

---

![Interactive Neural Network Playground Demo]
<img width="1863" height="919" alt="Screenshot 2025-11-11 234939" src="https://github.com/user-attachments/assets/f1437376-4e09-4b01-b6a5-b5b98c01ccb5" />
<img width="1842" height="911" alt="Screenshot 2025-11-11 235008" src="https://github.com/user-attachments/assets/7b9d075c-e379-4d42-9e1c-55a60c0de61e" />



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

## 🚀 How to Run on Google AI Studio 🚀

This project is designed to be run directly within Google AI Studio by uploading a `.zip` file. Follow these steps carefully to ensure the file structure is correct.

### Step 1: Download the Project from GitHub

1.  On the main page of this GitHub repository, click the green **`< > Code`** button.
2.  In the dropdown menu, select **`Download ZIP`**.
3.  Save the file (`interactive-neural-network-playground-main.zip` or similar) to your computer.

### Step 2: Prepare the ZIP for AI Studio

This is the most important step. AI Studio requires the `index.html` file to be at the top level of the zip file, but the GitHub download puts it inside a folder.

1.  **Unzip the downloaded file.** You will now have a folder named something like `interactive-neural-network-playground-main`.

2.  **Open the folder.** Navigate *inside* the `interactive-neural-network-playground-main` folder. You should see all the project files and folders (`index.html`, `App.tsx`, `components`, `services`, etc.).

3.  **Select the application files.** Select **all** the files and folders *inside* this directory that are needed for the app.
    -   **Include:** `App.tsx`, `components/`, `constants.ts`, `docs/`, `index.html`, `index.tsx`, `metadata.json`, `README.md`, `services/`, `types.ts`
    -   **Do not go back up and select the parent folder.** Stay inside `interactive-neural-network-playground-main`.

4.  **Create the new ZIP file.** With all the app files selected, right-click and choose:
    -   **Windows:** `Send to > Compressed (zipped) folder`.
    -   **Mac:** `Compress [X] items`.
    
5.  **Rename the new ZIP file** to something clear, like `aistudio-upload.zip`.

> **CRITICAL:** By zipping the contents directly, you ensure that `index.html` is at the root of your new zip file, which is what AI Studio needs.

### Step 3: Upload and Run in AI Studio

1.  **Go to Google AI Studio:** Open your web browser and navigate to [aistudio.google.com](https://aistudio.google.com).

2.  **Create a New Project:** Start a blank, new project.

3.  **Upload Your ZIP:** Find the option to upload files (often a button or a drag-and-drop area) and select the `aistudio-upload.zip` file you created in the previous step.

4.  **Run the App:** AI Studio will automatically unzip the files, build the project, and launch the application in the preview panel.

5.  **Interact:** You can now interact with the live application!
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
