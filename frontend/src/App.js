import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TrainingInterface = () => {
  const [trainingPairs, setTrainingPairs] = useState([]);
  const [newPair, setNewPair] = useState({ query: "", response: "", quality_score: 0.8 });
  const [isTraining, setIsTraining] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({
    epochs: 10,
    batch_size: 16,
    learning_rate: 0.0001
  });

  const loadTrainingData = async () => {
    try {
      const result = await axios.get(`${API}/training/data`);
      setTrainingPairs(result.data.training_pairs || []);
    } catch (err) {
      console.error("Failed to load training data:", err);
    }
  };

  const addTrainingPair = async () => {
    if (!newPair.query.trim() || !newPair.response.trim()) return;
    
    try {
      await axios.post(`${API}/training/add-pair`, {
        query: newPair.query,
        response: newPair.response,
        quality_score: newPair.quality_score,
        coherence_score: 0.8,
        sememes: []
      });
      
      setNewPair({ query: "", response: "", quality_score: 0.8 });
      loadTrainingData();
    } catch (err) {
      console.error("Failed to add training pair:", err);
    }
  };

  const startTraining = async () => {
    if (trainingPairs.length === 0) {
      alert("Please add some training pairs first!");
      return;
    }
    
    setIsTraining(true);
    
    try {
      const response = await axios.post(`${API}/training/start`, {
        training_pairs: trainingPairs,
        epochs: trainingConfig.epochs,
        batch_size: trainingConfig.batch_size,
        learning_rate: trainingConfig.learning_rate
      });
      
      alert(response.data.message);
    } catch (err) {
      console.error("Failed to start training:", err);
      alert("Training failed to start");
    } finally {
      setIsTraining(false);
    }
  };

  const clearTrainingData = async () => {
    if (confirm("Are you sure you want to clear all training data?")) {
      try {
        await axios.delete(`${API}/training/data`);
        loadTrainingData();
      } catch (err) {
        console.error("Failed to clear training data:", err);
      }
    }
  };

  useEffect(() => {
    loadTrainingData();
  }, []);

  return (
    <div className="training-interface">
      <h2>🎓 Training Interface</h2>
      
      {/* Training Configuration */}
      <div className="training-config">
        <h3>Training Configuration</h3>
        <div className="config-grid">
          <div className="config-item">
            <label>Epochs:</label>
            <input
              type="number"
              value={trainingConfig.epochs}
              onChange={(e) => setTrainingConfig({...trainingConfig, epochs: parseInt(e.target.value)})}
              min="1"
              max="100"
            />
          </div>
          <div className="config-item">
            <label>Batch Size:</label>
            <input
              type="number"
              value={trainingConfig.batch_size}
              onChange={(e) => setTrainingConfig({...trainingConfig, batch_size: parseInt(e.target.value)})}
              min="1"
              max="64"
            />
          </div>
          <div className="config-item">
            <label>Learning Rate:</label>
            <input
              type="number"
              value={trainingConfig.learning_rate}
              onChange={(e) => setTrainingConfig({...trainingConfig, learning_rate: parseFloat(e.target.value)})}
              step="0.0001"
              min="0.0001"
              max="0.01"
            />
          </div>
        </div>
      </div>

      {/* Add Training Pair */}
      <div className="add-training-pair">
        <h3>Add Training Pair</h3>
        <div className="pair-form">
          <textarea
            placeholder="Enter query..."
            value={newPair.query}
            onChange={(e) => setNewPair({...newPair, query: e.target.value})}
            rows={2}
          />
          <textarea
            placeholder="Enter expected response..."
            value={newPair.response}
            onChange={(e) => setNewPair({...newPair, response: e.target.value})}
            rows={3}
          />
          <div className="quality-score">
            <label>Quality Score:</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={newPair.quality_score}
              onChange={(e) => setNewPair({...newPair, quality_score: parseFloat(e.target.value)})}
            />
            <span>{newPair.quality_score.toFixed(1)}</span>
          </div>
          <button onClick={addTrainingPair} className="add-pair-btn">
            Add Training Pair
          </button>
        </div>
      </div>

      {/* Training Data */}
      <div className="training-data">
        <div className="data-header">
          <h3>Training Data ({trainingPairs.length} pairs)</h3>
          <div className="data-actions">
            <button onClick={startTraining} disabled={isTraining || trainingPairs.length === 0} className="train-btn">
              {isTraining ? "Training..." : "Start Training"}
            </button>
            <button onClick={clearTrainingData} className="clear-btn">
              Clear Data
            </button>
          </div>
        </div>
        
        <div className="training-pairs-list">
          {trainingPairs.map((pair, index) => (
            <div key={index} className="training-pair">
              <div className="pair-query">
                <strong>Q:</strong> {pair.query}
              </div>
              <div className="pair-response">
                <strong>A:</strong> {pair.response}
              </div>
              <div className="pair-scores">
                <span>Quality: {pair.quality_score.toFixed(1)}</span>
                <span>Coherence: {pair.coherence_score.toFixed(1)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const ResponseImprovement = ({ query, currentResponse, onImprove }) => {
  const [improvedResponse, setImprovedResponse] = useState("");
  const [isImproving, setIsImproving] = useState(false);

  const improveResponse = async () => {
    if (!improvedResponse.trim()) return;
    
    setIsImproving(true);
    
    try {
      const response = await axios.post(`${API}/training/improve-response`, null, {
        params: {
          query: query,
          current_response: currentResponse,
          target_response: improvedResponse
        }
      });
      
      onImprove(response.data);
      setImprovedResponse("");
    } catch (err) {
      console.error("Failed to improve response:", err);
    } finally {
      setIsImproving(false);
    }
  };

  return (
    <div className="response-improvement">
      <h4>💡 Improve This Response</h4>
      <div className="improvement-form">
        <textarea
          placeholder="Enter improved response..."
          value={improvedResponse}
          onChange={(e) => setImprovedResponse(e.target.value)}
          rows={3}
        />
        <button onClick={improveResponse} disabled={isImproving} className="improve-btn">
          {isImproving ? "Adding..." : "Add as Training"}
        </button>
      </div>
    </div>
  );
};

const CognitionInterface = () => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);
  const [showTraining, setShowTraining] = useState(false);
  const [showImprovement, setShowImprovement] = useState(false);

  // Load initial data
  useEffect(() => {
    loadHistory();
    loadMetrics();
  }, []);

  const loadHistory = async () => {
    try {
      const result = await axios.get(`${API}/cognition/history`);
      setHistory(result.data);
    } catch (err) {
      console.error("Failed to load history:", err);
    }
  };

  const loadMetrics = async () => {
    try {
      const result = await axios.get(`${API}/cognition/performance`);
      setMetrics(result.data);
    } catch (err) {
      console.error("Failed to load metrics:", err);
    }
  };

  const processQuery = async () => {
    if (!query.trim()) return;
    
    setIsProcessing(true);
    setError(null);
    
    try {
      const result = await axios.post(`${API}/cognition`, { query });
      setResponse(result.data);
      setQuery("");
      
      // Refresh history and metrics
      loadHistory();
      loadMetrics();
    } catch (err) {
      setError(err.response?.data?.detail || "Processing failed");
      console.error("Cognition processing failed:", err);
    } finally {
      setIsProcessing(false);
    }
  };

  const resetEngine = async () => {
    try {
      await axios.post(`${API}/cognition/reset`);
      setResponse(null);
      setHistory([]);
      setMetrics(null);
      loadMetrics();
      alert("Engine reset successfully!");
    } catch (err) {
      alert("Failed to reset engine");
    }
  };

  const handleResponseImprovement = (improvementData) => {
    alert("Response improvement added to training data!");
    setShowImprovement(false);
  };

  const formatTime = (seconds) => {
    return `${(seconds * 1000).toFixed(1)}ms`;
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className="cognition-interface">
      <div className="header">
        <h1>🧠 ATC Research System</h1>
        <p className="subtitle">Multi-Phase Cognitive Processing Prototype</p>
        <div className="header-actions">
          <button 
            onClick={() => setShowTraining(!showTraining)}
            className="training-toggle-btn"
          >
            {showTraining ? "Hide Training" : "Show Training"}
          </button>
        </div>
      </div>

      {/* Training Interface */}
      {showTraining && <TrainingInterface />}

      {/* Query Input */}
      <div className="query-section">
        <div className="input-group">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your query for the SATC cognition engine..."
            className="query-input"
            rows={3}
            disabled={isProcessing}
          />
          <div className="input-actions">
            <button 
              onClick={processQuery}
              disabled={isProcessing || !query.trim()}
              className="process-btn"
            >
              {isProcessing ? "Processing..." : "Process Query"}
            </button>
            <button onClick={resetEngine} className="reset-btn">
              Reset Engine
            </button>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-section">
          <h3>❌ Error</h3>
          <p>{error}</p>
        </div>
      )}

      {/* Response Display */}
      {response && (
        <div className="response-section">
          <h3>🎯 Response</h3>
          <div className="response-card">
            <div className="response-header">
              <span className={`phase-badge ${response.phase}`}>
                {response.phase.toUpperCase()}
              </span>
              <span className="method-badge">{response.method}</span>
              <span className="time-badge">{formatTime(response.processing_time)}</span>
              <button 
                onClick={() => setShowImprovement(!showImprovement)}
                className="improve-toggle-btn"
              >
                💡 Improve
              </button>
            </div>
            
            <div className="response-content">
              <p className="output">{response.output}</p>
            </div>
            
            <div className="response-metrics">
              <div className="metric">
                <span className="metric-label">Coherence:</span>
                <span className="metric-value coherence">
                  {(response.coherence * 100).toFixed(1)}%
                </span>
              </div>
              {response.dissonance && (
                <div className="metric">
                  <span className="metric-label">Dissonance:</span>
                  <span className="metric-value dissonance">
                    {response.dissonance.toFixed(3)}
                  </span>
                </div>
              )}
              <div className="metric">
                <span className="metric-label">Success:</span>
                <span className={`metric-value ${response.success ? 'success' : 'failure'}`}>
                  {response.success ? "✅" : "❌"}
                </span>
              </div>
            </div>

            {response.metadata && (
              <div className="metadata-section">
                <h4>Technical Details</h4>
                <div className="metadata-grid">
                  <div className="metadata-item">
                    <span>Nodes:</span>
                    <span>{response.metadata.nodes_count}</span>
                  </div>
                  <div className="metadata-item">
                    <span>Sememes:</span>
                    <span>{response.metadata.sememes_count}</span>
                  </div>
                  <div className="metadata-item">
                    <span>Variants:</span>
                    <span>{response.metadata.variants_count}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Response Improvement */}
            {showImprovement && (
              <ResponseImprovement 
                query={response.query}
                currentResponse={response.output}
                onImprove={handleResponseImprovement}
              />
            )}
          </div>
        </div>
      )}

      {/* Performance Metrics */}
      {metrics && (
        <div className="metrics-section">
          <h3>📊 Performance Metrics</h3>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-title">Total Queries</div>
              <div className="metric-number">{metrics.total_queries}</div>
            </div>
            <div className="metric-card">
              <div className="metric-title">Recognition Rate</div>
              <div className="metric-number">{formatPercentage(metrics.recognition_rate)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-title">Avg Coherence</div>
              <div className="metric-number">{formatPercentage(metrics.avg_coherence)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-title">Avg Processing Time</div>
              <div className="metric-number">{formatTime(metrics.avg_processing_time)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-title">Memory Updates</div>
              <div className="metric-number">{metrics.memory_updates}</div>
            </div>
            <div className="metric-card">
              <div className="metric-title">Deposited Patterns</div>
              <div className="metric-number">{metrics.deposited_patterns}</div>
            </div>
          </div>
        </div>
      )}

      {/* History Section */}
      {history.length > 0 && (
        <div className="history-section">
          <h3>📜 Recent History</h3>
          <div className="history-list">
            {history.slice(0, 5).map((item, index) => (
              <div key={item.id || index} className="history-item">
                <div className="history-header">
                  <span className={`phase-badge ${item.phase}`}>
                    {item.phase.toUpperCase()}
                  </span>
                  <span className="history-time">
                    {formatTime(item.processing_time)}
                  </span>
                  <span className="history-coherence">
                    {(item.coherence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="history-query">{item.query}</div>
                <div className="history-output">{item.output}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const Home = () => {
  const [showCognition, setShowCognition] = useState(false);

  const testBackendConnection = async () => {
    try {
      const response = await axios.get(`${API}/`);
      console.log("Backend connection:", response.data.message);
    } catch (e) {
      console.error("Backend connection error:", e);
    }
  };

  useEffect(() => {
    testBackendConnection();
  }, []);

  if (showCognition) {
    return <CognitionInterface />;
  }

  return (
    <div className="home-page">
      <header className="app-header">
        <div className="hero-section">
          <h1 className="hero-title">🧠 ATC Research System</h1>
          <p className="hero-subtitle">Artificial True Cognition Research Prototype</p>
          <p className="hero-description">
            Experimental multi-phase cognitive processing system combining established machine learning 
            techniques with research approaches toward artificial general intelligence.
          </p>
        </div>

        <div className="features-section">
          <h2>System Components</h2>
          <div className="features-grid">
            <div className="feature-card">
              <h3>🔗 Neural Processing</h3>
              <p>12-layer neural network with square dimensional progression (784→1)</p>
            </div>
            <div className="feature-card">
              <h3>⚡ Recognition System</h3>
              <p>FAISS-based pattern matching with automatic learning (0.01-0.1s)</p>
            </div>
            <div className="feature-card">
              <h3>🎯 Quality Optimization</h3>
              <p>Beam search and genetic algorithms for output optimization</p>
            </div>
            <div className="feature-card">
              <h3>🧠 BERT Embeddings</h3>
              <p>Real semantic processing via sentence-transformers</p>
            </div>
            <div className="feature-card">
              <h3>🔬 Experimental Phases</h3>
              <p>Research implementation of reflection, volition, and personality phases</p>
            </div>
            <div className="feature-card">
              <h3>📊 Performance Tracking</h3>
              <p>Comprehensive metrics and processing analytics</p>
            </div>
          </div>
        </div>

        <div className="cta-section">
          <button 
            onClick={() => setShowCognition(true)}
            className="cta-button"
          >
            🚀 Launch Research Interface
          </button>
          <p className="cta-description">
            Interact with the ATC research prototype
          </p>
        </div>
      </header>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <Home />
    </div>
  );
}

export default App;