/**
 * @file App.tsx
 * @description The root component of the Interactive Neural Network Playground.
 * It manages the main layout, header, and tab navigation to switch between
 * different playground modes and the interactive demo.
 */

import React, { useState } from 'react';
import { SimplePlayground } from './components/SimplePlayground';
import { AdvancedPlayground } from './components/AdvancedPlayground';
import { SuperAdvancedPlayground } from './components/SuperAdvancedPlayground';
import { InteractiveDemo } from './components/InteractiveDemo';
import { AboutPanel } from './components/AboutPanel';
import { LSTMPlayground } from './components/LSTMPlayground';

// Define the possible tabs the user can select. Each corresponds to a different model or view.
type Tab = 'simple' | 'advanced' | 'super-advanced' | 'lstm' | 'interactive';

// FIX: Extracted component props to a dedicated interface to resolve a compiler issue with inferring the 'children' prop.
interface TabButtonProps {
  tab: Tab;
  children: React.ReactNode;
}

const App = () => {
  // State to keep track of the currently active tab. Defaults to the interactive demo.
  const [activeTab, setActiveTab] = useState<Tab>('interactive');

  /**
   * Renders the main content area based on the currently active tab.
   * This function acts as a router to display the correct playground component.
   */
  const renderContent = () => {
    switch (activeTab) {
      case 'simple':
        return <SimplePlayground />;
      case 'advanced':
        return <AdvancedPlayground />;
      case 'super-advanced':
        return <SuperAdvancedPlayground />;
      case 'lstm':
        return <LSTMPlayground />;
      case 'interactive':
        return <InteractiveDemo />;
      default:
        // Fallback to the interactive demo if the state is invalid for any reason.
        return <InteractiveDemo />;
    }
  };

  /**
   * A reusable button component for tab navigation.
   * It handles click events to set the active tab and applies conditional styling
   * to indicate which tab is currently selected.
   * @param {TabButtonProps} props - The props for the component, including the target tab and its label.
   */
  // FIX: Updated component to use React.FC to resolve issue with 'children' prop type inference.
  const TabButton: React.FC<TabButtonProps> = ({ tab, children }) => (
    <button
      onClick={() => setActiveTab(tab)}
      className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-cyan-500 ${
        activeTab === tab
          ? 'bg-gray-800 text-cyan-400' // Active tab style
          : 'text-gray-400 hover:text-white hover:bg-gray-700/50' // Inactive tab style
      }`}
    >
      {children}
    </button>
  );

  return (
    <div className="bg-gray-900 text-white min-h-screen font-sans p-4 sm:p-6 lg:p-8">
      <div className="max-w-screen-xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-cyan-400 tracking-tight">
            Interactive Neural Network Playground
          </h1>
          <p className="mt-2 text-lg text-gray-400">
            Train a character-level language model from scratch in your browser. Simulating how a child learns phonotactics
          </p>
          <div className="mt-6">
            {/* The AboutPanel provides a collapsible overview of the application's features. */}
            <AboutPanel />
          </div>
        </header>

        <main>
            {/* Tab navigation container */}
            <div className="border-b border-gray-700 mb-6">
                <nav className="flex flex-wrap space-x-1" aria-label="Tabs">
                    <TabButton tab="interactive">Interactive Demo</TabButton>
                    <TabButton tab="simple">Simple (FFNN)</TabButton>
                    <TabButton tab="advanced">RNN</TabButton>
                    <TabButton tab="super-advanced">GRU</TabButton>
                    <TabButton tab="lstm">Advanced (LSTM)</TabButton>
                </nav>
            </div>
            {/* Renders the content for the selected tab */}
            {renderContent()}
        </main>
      </div>
    </div>
  );
};

export default App;
