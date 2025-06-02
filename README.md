# Embedding Space Folder

An interactive environment for testing clustering and manipulation approaches on text embeddings from semantic models.

## Overview

This dashboard provides a flexible testing environment for:
- Visualizing embeddings from semantic models (e.g., sentence transformers)
- Experimenting with different clustering algorithms and parameters
- Testing embedding space transformations
- Analyzing semantic relationships through interactive selection and highlighting

## Features

### Clustering Algorithms
- **K-Means**: Standard centroid-based clustering
- **DBSCAN**: Density-based, finds arbitrary shaped clusters
- **Hierarchical**: Bottom-up clustering with multiple linkage options
- **Mean Shift**: Finds modes in density distribution
- **Spectral**: Graph-based for non-convex clusters
- **Gaussian Mixture**: Soft clustering with probabilistic assignments

### Embedding Manipulation
- Principal component-based compression transformation
- Adjustable compression factor and iterations
- Choose to cluster on original or transformed space
- Real-time metrics showing effect of transformations

### Interactive Analysis
- 3D visualization with PCA or t-SNE projection
- Click to select/highlight sentences and their embeddings
- Consistent cluster coloring across all views
- Elbow plot for cluster count optimization
- Export results for further analysis

## Installation

```bash
pip install dash dash-bootstrap-components plotly numpy pandas scikit-learn
```

## Usage

```bash
python embedding_space_folder.py
```

Open browser to `http://localhost:8050`

### Input Format

Upload embeddings from any semantic model as JSON:
```json
[
  {
    "sentence": "First sentence text",
    "embedding": [0.1, -0.2, 0.3, ...]
  },
  {
    "sentence": "Second sentence text", 
    "embedding": [0.2, 0.1, -0.1, ...]
  }
]
```

## Parameters

### Clustering
- Algorithm selection with specific parameters per method
- Cluster on original or transformed embeddings
- Real-time cluster statistics

### Visualization
- **PCA**: Linear projection preserving global variance
- **t-SNE**: Non-linear projection preserving local structure

### Transformation
- **Compression Factor**: Strength of transformation (0.1-1.0)
- **Iterations**: Number of transformation applications
- **Power Iterations**: Eigenvector computation accuracy

## Use Cases

- Test clustering approaches for semantic search applications
- Evaluate embedding quality from different models
- Develop semantic highlighting strategies
- Explore relationships in document collections
- Prototype embedding-based features
