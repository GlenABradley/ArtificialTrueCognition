import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const CognitionInterface = () => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

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

  const formatTime = (seconds) => {
    return `${(seconds * 1000).toFixed(1)}ms`;
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className="cognition-interface">
      <div className="header">
        <h1>🧠 Enhanced SATC Engine</h1>
        <p className="subtitle">Artificial True Cognition System</p>
      </div>

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
          <h1 className="hero-title">🧠 Enhanced SATC</h1>
          <p className="hero-subtitle">Artificial True Cognition System</p>
          <p className="hero-description">
            Revolutionary cognitive architecture that achieves genuine artificial cognition 
            through emergent semantic processing and the Syncopation engine.
          </p>
        </div>

        <div className="features-section">
          <h2>Key Features</h2>
          <div className="features-grid">
            <div className="feature-card">
              <h3>🔄 Syncopation Engine</h3>
              <p>Multi-dimensional semantic resonance cascades (12D→24D→48D→96D)</p>
            </div>
            <div className="feature-card">
              <h3>⚡ Dual-Phase Processing</h3>
              <p>Recognition (fast) + Cognition (deliberate) architecture</p>
            </div>
            <div className="feature-card">
              <h3>🎯 Dissonance Balancing</h3>
              <p>Optimal output generation through beam search and genetic algorithms</p>
            </div>
            <div className="feature-card">
              <h3>🧠 Continual Learning</h3>
              <p>EWC-based memory integration without catastrophic forgetting</p>
            </div>
            <div className="feature-card">
              <h3>🔬 HD Space Processing</h3>
              <p>10,000+ dimensional semantic representation</p>
            </div>
            <div className="feature-card">
              <h3>📊 Performance Monitoring</h3>
              <p>Real-time metrics and coherence tracking</p>
            </div>
          </div>
        </div>

        <div className="cta-section">
          <button 
            onClick={() => setShowCognition(true)}
            className="cta-button"
          >
            🚀 Launch SATC Interface
          </button>
          <p className="cta-description">
            Experience true artificial cognition in action
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