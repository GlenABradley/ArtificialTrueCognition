# Syncopation Engine

## Overview
The Syncopation Engine is a core component of the Artificial True Cognition system, implementing advanced semantic processing using a 12-dimensional vector space. This engine is responsible for managing semantic particles, calculating similarities, and enabling flexible semantic search with axis-wise tolerance.

## Key Features

### 1. Semantic Particles
- 12-dimensional vector representation
- Rich metadata support
- Cluster-based organization
- Dynamic similarity calculation

### 2. Advanced Similarity Metrics
- Mahalanobis distance for dimensional sensitivity
- Attention-based similarity weighting
- Cluster-aware matching
- Customizable axis tolerances

### 3. Test Suite
- Comprehensive test coverage
- Detailed debug output
- Cluster separation validation
- Negative inference testing

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy

### Installation
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test with debug output
pytest tests/test_semantic_axis.py::TestSemanticField::test_name -v
```

## Architecture

### Core Components

#### SemanticField
Manages a collection of semantic particles and handles similarity searches.

#### SemanticParticle
Represents a single semantic unit with vector and metadata.

#### Attention Mechanisms
Implements dynamic weighting of semantic dimensions.

## Recent Improvements

### 1. Cluster Separation
- Implemented robust cluster separation logic
- Added special handling for similar clusters
- Enhanced similarity calculation with cluster-based boosting

### 2. Similarity Calculation
- Integrated Mahalanobis distance
- Added attention-based weighting
- Implemented cluster-based penalties

### 3. Debugging Tools
- Detailed similarity analysis
- Cluster visualization
- Performance metrics

## Known Issues

### 1. Negative Inference Space
- Currently returns too many matches for invalid blends
- Planned improvements:
  - Stronger penalties for incompatible clusters
  - Enhanced test cases
  - Improved similarity thresholds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[Specify License]
