import React, { ReactNode } from 'react';

// FIX: Using React.FC to correctly type the component props, including children.
// FIX: Explicitly added children to the props type.
const DetailSection: React.FC<{ title: string; children: ReactNode; }> = ({ title, children }) => (
  <section className="mb-6">
    <h3 className="text-lg font-semibold text-cyan-400 mb-2 border-b border-gray-600 pb-1">{title}</h3>
    <div className="text-gray-300 space-y-2 text-sm">
      {children}
    </div>
  </section>
);

// FIX: Using React.FC to correctly type the component props, including children.
// FIX: Explicitly added children to the props type.
const ListItem: React.FC<{ term: string; children: ReactNode; }> = ({ term, children }) => (
    <li className="ml-4 list-disc list-outside marker:text-cyan-400">
        <strong className="text-gray-100">{term}:</strong> {children}
    </li>
);

export const AboutPanel = () => {
  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700">
      <details className="group">
        <summary className="p-4 cursor-pointer font-semibold text-cyan-400 list-none flex justify-between items-center">
          About This App & How It Works
          <span className="text-gray-400 transform transition-transform duration-200 group-open:rotate-180">▼</span>
        </summary>
        <div className="p-6 border-t border-gray-700">
            <DetailSection title="What is this?">
                <p>
                    This application is an interactive, educational tool designed to demystify the basics of how different neural networks learn to generate text. You are training a character-level language model from scratch, right in your browser. Its only goal is to look at one token (a character or sub-word) and predict the very next one in a sequence.
                </p>
                <p>
                    The app is split into five tabs, each offering a unique perspective:
                </p>
                 <ul className="space-y-2">
                    <ListItem term="Simple (FFNN)">
                        Uses a basic <strong>Feed-Forward Neural Network</strong>. This model has no memory; it makes predictions based only on the single, immediate input token. It's the best place to start to understand the fundamentals.
                    </ListItem>
                     <ListItem term="RNN">
                        Introduces a <strong>Recurrent Neural Network</strong>. This model has a simple form of memory (a "hidden state") that is updated with each step, allowing it to learn from sequences of tokens.
                    </ListItem>
                     <ListItem term="GRU">
                       Explores the <strong>Gated Recurrent Unit</strong>, an advanced RNN. It uses "gates" to more intelligently decide what information to remember or forget, helping it learn longer-range patterns.
                    </ListItem>
                     <ListItem term="Advanced (LSTM)">
                        Features a <strong>Long Short-Term Memory</strong> network, the most complex model here. LSTMs have a sophisticated gating mechanism and a separate "cell state" for memory, making them excellent at capturing long-term dependencies in text.
                    </ListItem>
                     <ListItem term="Interactive Demo">
                        A simplified, step-by-step visualization of the FFNN training process. It uses a tiny dataset to clearly show how weights and activations change with each training example.
                    </ListItem>
                </ul>
            </DetailSection>

            <DetailSection title="Key Controls & Hyperparameters">
                <ul>
                    <ListItem term="Training Text">
                        This is the source material the model learns from. It analyzes this text to learn which tokens tend to follow others.
                    </ListItem>
                     <ListItem term="Tokenizer">
                        Determines how the text is broken into "tokens" (the vocabulary). You can choose from simple characters, a learned sub-word vocabulary (BPE), or a custom list of tokens.
                    </ListItem>
                    <ListItem term="Learning Rate">
                        Controls how much the model adjusts its internal connections (weights) after each mistake. A high rate learns fast but can be unstable; a low rate is slow but more precise.
                    </ListItem>
                    <ListItem term="Hidden Size">
                        Determines the model's "brainpower." A larger size allows it to learn more complex patterns but requires more computation and can take longer to train.
                    </ListItem>
                    <ListItem term="Max Epochs">
                        One "epoch" is one full pass through the entire training text. Training will automatically stop if performance plateaus, even if this number isn't reached.
                    </ListItem>
                     <ListItem term="Batch Size / Sequence Length">
                        For the FFNN, this is the number of token pairs processed before updating weights. For RNN, GRU, and LSTM, this is the <strong>sequence length</strong> the model "unrolls" to learn from at each step.
                    </ListItem>
                    <ListItem term="Dropout Rate">
                        (RNN/GRU/LSTM only) A technique to prevent overfitting. During training, it randomly ignores a fraction of neurons, forcing the network to learn more robust patterns.
                    </ListItem>
                </ul>
            </DetailSection>
            
            <DetailSection title="Understanding the Visualizations">
                 <ul>
                    <ListItem term="Architecture Weights">
                       Visualizes all the weight and bias matrices in the current model as heatmaps. Brighter cyan values are positive, brighter red values are negative. This shows you the "brain" of the model as it learns.
                    </ListItem>
                    <ListItem term="Training Log">
                        A real-time feed showing the model's progress, its current error rate (Loss), and the effective learning rate (LR).
                    </ListItem>
                    <ListItem term="Generation & Coaching">
                        After training starts, you can use the model to generate words. If you get a good result, you can use the "Good 👍" button to reinforce that word by training the model on it for a few extra steps.
                    </ListItem>
                    <ListItem term="Generation History">
                        At key milestones, the model is asked to generate words. This panel collects them so you can see how its creativity and coherence improve over time.
                    </ListItem>
                    <ListItem term="Loss Histogram">
                        This chart shows the average loss (error) at the end of each epoch. A healthy training run will show a consistent downward trend.
                    </ListItem>
                    <ListItem term="Save / Load Model">
                        Allows you to save the trained state of your model to a file and load it back in later to continue training or generating text.
                    </ListItem>
                </ul>
            </DetailSection>

             <DetailSection title="Architectures & Training Features">
                 <ul>
                    <ListItem term="Early Stopping">
                        The app monitors the training loss. If the loss stops improving for several epochs, it assumes the model has learned as much as it can and automatically begins a "fine-tuning" phase.
                    </ListItem>
                    <ListItem term="Learning Rate Decay">
                        During the fine-tuning phase, the learning rate is gradually reduced, allowing the model to make smaller, more precise adjustments to its weights for better results.
                    </ListItem>
                    <ListItem term="Automatic Fast Mode">
                        To speed up long training runs, the app will automatically suspend most UI updates after 60 seconds. The visualizations will reappear for the final epoch so you can see the end result.
                    </ListItem>
                     <ListItem term="Cyclical Training & Auto-Coaching">
                        Automates the process of training and reinforcement. The app can run training cycles for a set number of epochs, then spend time auto-generating words, validating them against phonotactic rules, and automatically reinforcing the "good" ones. This creates a feedback loop that helps the model discover and strengthen its understanding of valid word structures.
                    </ListItem>
                    <ListItem term="The 'Good Word' Validator">
                        The Auto-Coach feature relies on a sophisticated validator to determine if a generated word is "good". This validator uses a set of rules based on English phonotactics to filter for plausible-sounding words. A word is considered "good" if it passes all the following checks:
                        <ul className="list-decimal list-inside mt-2 space-y-1">
                            <li><strong>Plausible Length:</strong> The word must be between 2 and 12 characters long.</li>
                            <li><strong>Has Vowels:</strong> The word must contain at least one vowel (a, e, i, o, u).</li>
                            <li><strong>Valid Consonant Clusters:</strong> It rejects words with three or more consonants in a row that don't form a valid cluster (e.g., "bfr" is bad, but "str" is good).</li>
                            <li><strong>No Unnatural Repetition:</strong> It filters out words with repeating patterns like "rererer" or "kalakala".</li>
                            <li><strong>Follows Syllable Structure:</strong> The word must be parsable into valid syllable components (Onset, Vowel, Coda) in a logical sequence.</li>
                        </ul>
                    </ListItem>
                 </ul>
            </DetailSection>
        </div>
      </details>
    </div>
  );
};