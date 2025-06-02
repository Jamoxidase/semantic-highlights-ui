import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import json
import base64
import io
from typing import List, Dict, Set, Tuple, Optional

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define cluster colors matching the React version
CLUSTER_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
                  '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#f1c40f']

class EmbeddingSpaceFolder:
    """Core computation class for embedding space operations"""
    
    def __init__(self):
        self.original_points = None
        self.transformed_points = None
        self.sentences = []
        self.selected_indices = set()
        self.clusters = None
        self.dimensions = None
        self.pca_components = 3  # For 3D visualization
        
    def load_embeddings(self, contents, filename):
        """Load embeddings from uploaded JSON file"""
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        try:
            data = json.loads(decoded.decode('utf-8'))
            self.sentences = [item['sentence'] for item in data]
            embeddings = np.array([item['embedding'] for item in data])
            self.original_points = embeddings
            self.transformed_points = embeddings.copy()
            self.dimensions = embeddings.shape[1]
            return True, f"Loaded {len(self.sentences)} embeddings with {self.dimensions} dimensions"
        except Exception as e:
            return False, f"Error loading file: {str(e)}"
    
    def generate_random_embeddings(self, n_points=50, n_dims=10):
        """Generate random embeddings for testing"""
        np.random.seed(42)
        self.original_points = np.random.randn(n_points, n_dims) * 10
        self.transformed_points = self.original_points.copy()
        self.sentences = [f"Random point {i}" for i in range(n_points)]
        self.dimensions = n_dims
        
    def fold_space(self, iterations=1, compression_factor=0.6, power_iterations=50):
        """Apply folding transformation"""
        if len(self.selected_indices) < 2:
            return self.transformed_points
            
        self.transformed_points = self.original_points.copy()
        
        for _ in range(iterations):
            selected_points = self.transformed_points[list(self.selected_indices)]
            centroid = np.mean(selected_points, axis=0)
            centered = selected_points - centroid
            cov_matrix = np.cov(centered.T)
            
            # Power iteration for principal eigenvector
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            fold_axis = eigenvectors[:, -1]  # Eigenvector with largest eigenvalue
            
            for i in range(len(self.transformed_points)):
                offset_from_centroid = self.transformed_points[i] - centroid
                projection_on_axis = np.dot(offset_from_centroid, fold_axis)
                self.transformed_points[i] -= projection_on_axis * fold_axis * compression_factor
                
        return self.transformed_points
    
    def compute_projection(self, method='pca', perplexity=15, n_iter=500):
        """Compute dimensionality reduction for 3D visualization"""
        if method == 'pca':
            pca_orig = PCA(n_components=3)
            pca_trans = PCA(n_components=3)
            orig_proj = pca_orig.fit_transform(self.original_points)
            trans_proj = pca_trans.fit_transform(self.transformed_points)
            
            # Calculate explained variance
            orig_var = pca_orig.explained_variance_ratio_
            trans_var = pca_trans.explained_variance_ratio_
            
            return orig_proj, trans_proj, orig_var, trans_var
        else:  # tsne
            # For t-SNE, we'll do 2D and add a zero z-coordinate
            tsne_orig = TSNE(n_components=2, perplexity=min(perplexity, len(self.original_points)-1), 
                            n_iter=n_iter, random_state=42)
            tsne_trans = TSNE(n_components=2, perplexity=min(perplexity, len(self.transformed_points)-1), 
                             n_iter=n_iter, random_state=42)
            orig_2d = tsne_orig.fit_transform(self.original_points)
            trans_2d = tsne_trans.fit_transform(self.transformed_points)
            
            # Add zero z-coordinate for 3D plot
            orig_proj = np.column_stack([orig_2d, np.zeros(len(orig_2d))])
            trans_proj = np.column_stack([trans_2d, np.zeros(len(trans_2d))])
            
            return orig_proj, trans_proj, None, None
    
    def perform_clustering(self, algorithm='kmeans', n_clusters=3, use_transformed=False, **kwargs):
        """Perform clustering with various algorithms"""
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        
        points = self.transformed_points if use_transformed else self.original_points
        
        # Standardize features for distance-based algorithms
        scaler = StandardScaler()
        scaled_points = scaler.fit_transform(points)
        
        if algorithm == 'kmeans':
            actual_clusters = min(n_clusters, len(points))
            clusterer = KMeans(n_clusters=actual_clusters, init='k-means++', n_init=10, random_state=42)
            self.clusters = clusterer.fit_predict(points)
            
        elif algorithm == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            self.clusters = clusterer.fit_predict(scaled_points)
            
        elif algorithm == 'hierarchical':
            linkage = kwargs.get('linkage', 'ward')
            actual_clusters = min(n_clusters, len(points))
            clusterer = AgglomerativeClustering(n_clusters=actual_clusters, linkage=linkage)
            self.clusters = clusterer.fit_predict(points)
            
        elif algorithm == 'meanshift':
            bandwidth = kwargs.get('bandwidth', None)
            clusterer = MeanShift(bandwidth=bandwidth, random_state=42)
            self.clusters = clusterer.fit_predict(scaled_points)
            
        elif algorithm == 'spectral':
            actual_clusters = min(n_clusters, len(points))
            clusterer = SpectralClustering(n_clusters=actual_clusters, random_state=42, n_jobs=-1)
            self.clusters = clusterer.fit_predict(points)
            
        elif algorithm == 'gmm':
            actual_clusters = min(n_clusters, len(points))
            clusterer = GaussianMixture(n_components=actual_clusters, random_state=42)
            self.clusters = clusterer.fit_predict(points)
            
        # Convert -1 labels (noise in DBSCAN) to a separate cluster
        if algorithm == 'dbscan' and -1 in self.clusters:
            max_cluster = max(self.clusters)
            self.clusters = [c if c != -1 else max_cluster + 1 for c in self.clusters]
            
        return self.clusters
    
    def compute_elbow_data(self, max_k=10, use_transformed=False):
        """Compute WCSS for elbow plot"""
        points = self.transformed_points if use_transformed else self.original_points
        wcss = []
        k_range = range(1, min(max_k + 1, len(points) // 2))
        
        saved_assignments = {}
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            assignments = kmeans.fit_predict(points)
            saved_assignments[k] = assignments
            wcss.append(kmeans.inertia_)
            
        return list(k_range), wcss, saved_assignments
    
    def calculate_compression_metric(self):
        """Calculate compression ratio"""
        if len(self.selected_indices) < 2:
            return 1.0
            
        selected_idx = list(self.selected_indices)
        original_dists = []
        transformed_dists = []
        
        for i in range(len(selected_idx)):
            for j in range(i + 1, len(selected_idx)):
                orig_dist = np.linalg.norm(
                    self.original_points[selected_idx[i]] - 
                    self.original_points[selected_idx[j]]
                )
                trans_dist = np.linalg.norm(
                    self.transformed_points[selected_idx[i]] - 
                    self.transformed_points[selected_idx[j]]
                )
                original_dists.append(orig_dist)
                transformed_dists.append(trans_dist)
                
        avg_orig = np.mean(original_dists)
        avg_trans = np.mean(transformed_dists)
        
        return avg_trans / avg_orig if avg_orig > 0 else 1.0

# Global instance
folder = EmbeddingSpaceFolder()

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Embedding Space Folder", className="mb-4"),
        ], width=12)
    ]),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Upload(
                                id='upload-data',
                                children=dbc.Button("Upload JSON", color="primary", size="sm"),
                                multiple=False
                            ),
                        ], width="auto"),
                        
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Dims:", style={"fontSize": "0.8rem"}),
                                dbc.Input(id="dims-input", type="number", value=10, min=2, max=50, size="sm"),
                            ], size="sm"),
                        ], width="auto"),
                        
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Points:", style={"fontSize": "0.8rem"}),
                                dbc.Input(id="points-input", type="number", value=50, min=10, max=200, size="sm"),
                            ], size="sm"),
                        ], width="auto"),
                        
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Algorithm:", style={"fontSize": "0.8rem"}),
                                dbc.Select(
                                    id="cluster-algorithm-select",
                                    options=[
                                        {"label": "K-Means", "value": "kmeans"},
                                        {"label": "DBSCAN", "value": "dbscan"},
                                        {"label": "Hierarchical", "value": "hierarchical"},
                                        {"label": "Mean Shift", "value": "meanshift"},
                                        {"label": "Spectral", "value": "spectral"},
                                        {"label": "Gaussian Mixture", "value": "gmm"}
                                    ],
                                    value="kmeans",
                                    size="sm"
                                ),
                            ], size="sm"),
                        ], width="auto"),
                        
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Clusters:", style={"fontSize": "0.8rem"}),
                                dbc.Input(id="clusters-input", type="number", value=3, min=1, max=10, size="sm"),
                            ], size="sm"),
                        ], width="auto"),
                        
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Cluster on:", style={"fontSize": "0.8rem"}),
                                dbc.Select(
                                    id="cluster-on-select",
                                    options=[
                                        {"label": "Original", "value": "original"},
                                        {"label": "Folded", "value": "transformed"}
                                    ],
                                    value="original",
                                    size="sm"
                                ),
                            ], size="sm"),
                        ], width="auto"),
                        
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Projection:", style={"fontSize": "0.8rem"}),
                                dbc.Select(
                                    id="projection-select",
                                    options=[
                                        {"label": "PCA", "value": "pca"},
                                        {"label": "t-SNE", "value": "tsne"}
                                    ],
                                    value="pca",
                                    size="sm"
                                ),
                            ], size="sm"),
                        ], width="auto"),
                    ], className="mb-2"),
                    
                    # Clustering algorithm-specific parameters
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.InputGroupText("Epsilon:", style={"fontSize": "0.8rem"}),
                                    dbc.Input(id="dbscan-eps-input", type="number", value=0.5, min=0.1, max=5.0, step=0.1, size="sm"),
                                ], size="sm"),
                            ], width="auto"),
                            
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.InputGroupText("Min Samples:", style={"fontSize": "0.8rem"}),
                                    dbc.Input(id="dbscan-minsamples-input", type="number", value=5, min=2, max=20, size="sm"),
                                ], size="sm"),
                            ], width="auto"),
                        ])
                    ], id="dbscan-params", style={"display": "none", "marginTop": "10px"}),
                    
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.InputGroupText("Linkage:", style={"fontSize": "0.8rem"}),
                                    dbc.Select(
                                        id="hierarchical-linkage-select",
                                        options=[
                                            {"label": "Ward", "value": "ward"},
                                            {"label": "Complete", "value": "complete"},
                                            {"label": "Average", "value": "average"},
                                            {"label": "Single", "value": "single"}
                                        ],
                                        value="ward",
                                        size="sm"
                                    ),
                                ], size="sm"),
                            ], width="auto"),
                        ])
                    ], id="hierarchical-params", style={"display": "none", "marginTop": "10px"}),
                    
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.InputGroupText("Bandwidth:", style={"fontSize": "0.8rem"}),
                                    dbc.Input(id="meanshift-bandwidth-input", type="number", placeholder="Auto", min=0.1, max=10.0, step=0.1, size="sm"),
                                ], size="sm"),
                            ], width="auto"),
                        ])
                    ], id="meanshift-params", style={"display": "none", "marginTop": "10px"}),
                    
                    # t-SNE parameters row
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.InputGroupText("Perplexity:", style={"fontSize": "0.8rem"}),
                                    dbc.Input(id="perplexity-input", type="number", value=15, min=5, max=50, size="sm"),
                                ], size="sm"),
                            ], width="auto"),
                            
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.InputGroupText("t-SNE Iter:", style={"fontSize": "0.8rem"}),
                                    dbc.Input(id="tsne-iter-input", type="number", value=500, min=100, max=2000, step=100, size="sm"),
                                ], size="sm"),
                            ], width="auto"),
                        ])
                    ], id="tsne-params", style={"display": "none", "marginTop": "10px"}),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Fold Iterations:", style={"fontSize": "0.8rem"}),
                                dbc.Input(id="fold-iter-input", type="number", value=1, min=0, max=10, size="sm"),
                            ], size="sm"),
                        ], width="auto"),
                        
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Compression:", style={"fontSize": "0.8rem"}),
                                dbc.Input(id="compression-input", type="number", value=0.6, min=0.1, max=1.0, step=0.1, size="sm"),
                            ], size="sm"),
                        ], width="auto"),
                        
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Power Iter:", style={"fontSize": "0.8rem"}),
                                dbc.Input(id="power-iter-input", type="number", value=50, min=10, max=100, size="sm"),
                            ], size="sm"),
                        ], width="auto"),
                        
                        dbc.Col([
                            dbc.Button("Generate Random", id="generate-btn", color="info", size="sm", className="me-2"),
                            dbc.Button("Clear Selection", id="clear-selection-btn", color="danger", size="sm", className="me-2"),
                            dbc.Button("Show Elbow", id="elbow-btn", color="secondary", size="sm"),
                        ], width="auto"),
                    ], className="mt-2"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="status-message", className="mt-2"),
                            html.Div(id="compression-metric", className="mt-2 text-success fw-bold"),
                        ], width=12),
                    ]),
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    # 3D Scatter Plots with stats
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Original Embedding Space", className="mb-0"),
                    html.Small(id="original-stats", className="text-muted")
                ]),
                dbc.CardBody([
                    dcc.Graph(id='original-scatter', config={'displayModeBar': True})
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Folded Embedding Space", className="mb-0"),
                    html.Small(id="transformed-stats", className="text-muted")
                ]),
                dbc.CardBody([
                    dcc.Graph(id='transformed-scatter', config={'displayModeBar': True})
                ])
            ])
        ], width=6),
    ], className="mb-4"),
    
    # Elbow Plot
    dbc.Row([
        dbc.Col([
            dbc.Collapse([
                dbc.Card([
                    dbc.CardHeader("Elbow Method Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id='elbow-plot', config={'displayModeBar': False})
                    ])
                ])
            ], id="elbow-collapse", is_open=False)
        ], width=12)
    ], className="mb-4"),
    
    # Cluster Analysis Panel
    dbc.Row([
        dbc.Col([
            html.Div(id="cluster-analysis", className="mb-4")
        ], width=12)
    ]),
    
    # Sentences Display
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    "Sentences - Click to Select/Deselect",
                    html.Div(id="sentence-stats", className="float-end text-muted", style={"fontSize": "0.9rem"})
                ]),
                dbc.CardBody([
                    html.Div(id="sentences-display", style={"maxHeight": "400px", "overflowY": "auto"})
                ])
            ])
        ], width=12)
    ]),
    
    # Hidden stores
    dcc.Store(id='selected-points-store', data=[]),
    dcc.Store(id='data-store', data={}),
    dcc.Store(id='saved-cluster-assignments', data={}),
], fluid=True)

# Callback to show/hide clustering algorithm parameters
@app.callback(
    [Output('dbscan-params', 'style'),
     Output('hierarchical-params', 'style'),
     Output('meanshift-params', 'style'),
     Output('clusters-input', 'disabled')],
    Input('cluster-algorithm-select', 'value')
)
def toggle_cluster_params(algorithm):
    dbscan_style = {"display": "block", "marginTop": "10px"} if algorithm == 'dbscan' else {"display": "none"}
    hierarchical_style = {"display": "block", "marginTop": "10px"} if algorithm == 'hierarchical' else {"display": "none"}
    meanshift_style = {"display": "block", "marginTop": "10px"} if algorithm == 'meanshift' else {"display": "none"}
    
    # Disable clusters input for algorithms that determine clusters automatically
    clusters_disabled = algorithm in ['dbscan', 'meanshift']
    
    return dbscan_style, hierarchical_style, meanshift_style, clusters_disabled

# Callback to show/hide t-SNE parameters
@app.callback(
    Output('tsne-params', 'style'),
    Input('projection-select', 'value')
)
def toggle_tsne_params(projection):
    if projection == 'tsne':
        return {"display": "block", "marginTop": "10px"}
    return {"display": "none"}

# Main data loading callback
@app.callback(
    [Output('data-store', 'data'),
     Output('status-message', 'children')],
    [Input('upload-data', 'contents'),
     Input('generate-btn', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('dims-input', 'value'),
     State('points-input', 'value')]
)
def update_data(contents, generate_clicks, filename, dims, points):
    ctx = callback_context
    
    if not ctx.triggered:
        folder.generate_random_embeddings(50, 10)
        return {'generated': True}, "Generated 50 random points with 10 dimensions"
    
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger == 'upload-data' and contents:
        success, message = folder.load_embeddings(contents, filename)
        return {'uploaded': success}, message
    
    elif trigger == 'generate-btn':
        folder.generate_random_embeddings(points, dims)
        return {'generated': True}, f"Generated {points} random points with {dims} dimensions"
    
    return {}, ""

# Selection callback - handles both sentence clicks and scatter plot clicks
@app.callback(
    Output('selected-points-store', 'data'),
    [Input('clear-selection-btn', 'n_clicks'),
     Input({'type': 'sentence-div', 'index': ALL}, 'n_clicks'),
     Input('original-scatter', 'clickData'),
     Input('transformed-scatter', 'clickData')],
    [State('selected-points-store', 'data')]
)
def update_selection(clear_clicks, sentence_clicks, orig_click, trans_click, current_selection):
    ctx = callback_context
    
    if not ctx.triggered:
        return []
    
    trigger = ctx.triggered[0]['prop_id']
    
    if 'clear-selection-btn' in trigger:
        folder.selected_indices = set()
        return []
    
    # Handle sentence click
    if 'sentence-div' in trigger:
        clicked_id = json.loads(trigger.split('.')[0])['index']
        current_set = set(current_selection)
        
        if clicked_id in current_set:
            current_set.remove(clicked_id)
        else:
            current_set.add(clicked_id)
        
        folder.selected_indices = current_set
        return list(current_set)
    
    # Handle scatter plot clicks
    if 'scatter' in trigger and (orig_click or trans_click):
        click_data = orig_click if 'original' in trigger else trans_click
        if click_data and 'points' in click_data:
            # For 3D plots, the index is in 'pointNumber' not 'pointIndex'
            clicked_idx = click_data['points'][0].get('pointNumber', click_data['points'][0].get('pointIndex', None))
            if clicked_idx is not None:
                current_set = set(current_selection)
                
                if clicked_idx in current_set:
                    current_set.remove(clicked_idx)
                else:
                    current_set.add(clicked_idx)
                
                folder.selected_indices = current_set
                return list(current_set)
    
    return current_selection

# Main plotting callback
@app.callback(
    [Output('original-scatter', 'figure'),
     Output('transformed-scatter', 'figure'),
     Output('compression-metric', 'children'),
     Output('original-stats', 'children'),
     Output('transformed-stats', 'children')],
    [Input('data-store', 'data'),
     Input('selected-points-store', 'data'),
     Input('fold-iter-input', 'value'),
     Input('compression-input', 'value'),
     Input('power-iter-input', 'value'),
     Input('cluster-algorithm-select', 'value'),
     Input('clusters-input', 'value'),
     Input('cluster-on-select', 'value'),
     Input('projection-select', 'value'),
     Input('perplexity-input', 'value'),
     Input('tsne-iter-input', 'value'),
     Input('dbscan-eps-input', 'value'),
     Input('dbscan-minsamples-input', 'value'),
     Input('hierarchical-linkage-select', 'value'),
     Input('meanshift-bandwidth-input', 'value')]
)
def update_plots(data, selected_points, fold_iter, compression, power_iter, cluster_algorithm,
                n_clusters, cluster_on, projection, perplexity, tsne_iter,
                dbscan_eps, dbscan_minsamples, hierarchical_linkage, meanshift_bandwidth):
    if folder.original_points is None:
        return {}, {}, "", "", ""
    
    # Apply folding
    folder.fold_space(fold_iter, compression, power_iter)
    
    # Perform clustering with selected algorithm
    use_transformed = cluster_on == 'transformed'
    
    # Build kwargs for clustering algorithms
    cluster_kwargs = {}
    if cluster_algorithm == 'dbscan':
        cluster_kwargs = {'eps': dbscan_eps, 'min_samples': dbscan_minsamples}
    elif cluster_algorithm == 'hierarchical':
        cluster_kwargs = {'linkage': hierarchical_linkage}
    elif cluster_algorithm == 'meanshift':
        if meanshift_bandwidth:
            cluster_kwargs = {'bandwidth': meanshift_bandwidth}
    
    folder.perform_clustering(cluster_algorithm, n_clusters, use_transformed, **cluster_kwargs)
    
    # Compute projections
    orig_proj, trans_proj, orig_var, trans_var = folder.compute_projection(projection, perplexity, tsne_iter)
    
    # Create color array based on clusters
    if folder.clusters is not None:
        colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in folder.clusters]
    else:
        colors = ['blue'] * len(folder.sentences)
    
    # Create 3D scatter plots
    fig_orig = go.Figure()
    fig_trans = go.Figure()
    
    # Add main scatter points
    fig_orig.add_trace(go.Scatter3d(
        x=orig_proj[:, 0],
        y=orig_proj[:, 1],
        z=orig_proj[:, 2] if projection == 'pca' else np.zeros(len(orig_proj)),
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            line=dict(width=0.5, color='white')
        ),
        text=[f"Point {i}: {sent[:50]}..." for i, sent in enumerate(folder.sentences)],
        hovertemplate='%{text}<br>%{x:.3f}, %{y:.3f}, %{z:.3f}<extra></extra>',
        showlegend=False
    ))
    
    fig_trans.add_trace(go.Scatter3d(
        x=trans_proj[:, 0],
        y=trans_proj[:, 1],
        z=trans_proj[:, 2] if projection == 'pca' else np.zeros(len(trans_proj)),
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            line=dict(width=0.5, color='white')
        ),
        text=[f"Point {i}: {sent[:50]}..." for i, sent in enumerate(folder.sentences)],
        hovertemplate='%{text}<br>%{x:.3f}, %{y:.3f}, %{z:.3f}<extra></extra>',
        showlegend=False
    ))
    
    # Highlight selected points with ring instead of covering them
    if selected_points:
        selected_idx = list(selected_points)
        selected_colors = [CLUSTER_COLORS[folder.clusters[i] % len(CLUSTER_COLORS)] if folder.clusters is not None and i < len(folder.clusters) else 'blue' for i in selected_idx]
        
        fig_orig.add_trace(go.Scatter3d(
            x=orig_proj[selected_idx, 0],
            y=orig_proj[selected_idx, 1],
            z=orig_proj[selected_idx, 2] if projection == 'pca' else np.zeros(len(selected_idx)),
            mode='markers',
            marker=dict(
                size=12,
                color=selected_colors,
                line=dict(color='black', width=4),
                symbol='circle-open'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_trans.add_trace(go.Scatter3d(
            x=trans_proj[selected_idx, 0],
            y=trans_proj[selected_idx, 1],
            z=trans_proj[selected_idx, 2] if projection == 'pca' else np.zeros(len(selected_idx)),
            mode='markers',
            marker=dict(
                size=12,
                color=selected_colors,
                line=dict(color='black', width=4),
                symbol='circle-open'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layouts
    if projection == 'pca':
        axis_labels = ['PC1', 'PC2', 'PC3']
    else:
        axis_labels = ['t-SNE 1', 't-SNE 2', '']
    
    scene_dict = dict(
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1],
        zaxis_title=axis_labels[2] if projection == 'pca' else '',
        bgcolor="rgb(250, 250, 250)",
        xaxis=dict(gridcolor='rgb(230, 230, 230)'),
        yaxis=dict(gridcolor='rgb(230, 230, 230)'),
        zaxis=dict(gridcolor='rgb(230, 230, 230)' if projection == 'pca' else dict(showgrid=False, showticklabels=False)),
    )
    
    fig_orig.update_layout(
        scene=scene_dict,
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision='constant'  # Preserve camera angle
    )
    
    fig_trans.update_layout(
        scene=scene_dict,
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        uirevision='constant'  # Preserve camera angle
    )
    
    # Calculate stats
    orig_stats = f"n={len(orig_proj)}, Selected: {len(selected_points)}"
    trans_stats = f"n={len(trans_proj)}, Selected: {len(selected_points)}"
    
    if projection == 'pca' and orig_var is not None:
        orig_stats += f", Var: {orig_var[0]:.1%}, {orig_var[1]:.1%}, {orig_var[2]:.1%}"
        trans_stats += f", Var: {trans_var[0]:.1%}, {trans_var[1]:.1%}, {trans_var[2]:.1%}"
    
    # Calculate compression metric
    compression_ratio = folder.calculate_compression_metric()
    compression_text = f"Compression: {compression_ratio:.3f}x" if len(selected_points) >= 2 else ""
    
    return fig_orig, fig_trans, compression_text, orig_stats, trans_stats

@app.callback(
    [Output('sentences-display', 'children'),
     Output('sentence-stats', 'children')],
    [Input('data-store', 'data'),
     Input('selected-points-store', 'data'),
     Input('cluster-algorithm-select', 'value'),
     Input('clusters-input', 'value'),
     Input('cluster-on-select', 'value'),
     Input('fold-iter-input', 'value'),
     Input('compression-input', 'value'),
     Input('dbscan-eps-input', 'value'),
     Input('dbscan-minsamples-input', 'value'),
     Input('hierarchical-linkage-select', 'value'),
     Input('meanshift-bandwidth-input', 'value'),
     Input('saved-cluster-assignments', 'data')]
)
def update_sentences_display(data, selected_points, cluster_algorithm, n_clusters, cluster_on,
                           fold_iter, compression, dbscan_eps, dbscan_minsamples, 
                           hierarchical_linkage, meanshift_bandwidth, saved_assignments):
    if not folder.sentences:
        return [], ""
    
    sentence_divs = []
    
    # Make sure clustering is up to date with current parameters
    if folder.clusters is None or len(folder.clusters) != len(folder.sentences):
        use_transformed = cluster_on == 'transformed'
        cluster_kwargs = {}
        if cluster_algorithm == 'dbscan':
            cluster_kwargs = {'eps': dbscan_eps, 'min_samples': dbscan_minsamples}
        elif cluster_algorithm == 'hierarchical':
            cluster_kwargs = {'linkage': hierarchical_linkage}
        elif cluster_algorithm == 'meanshift' and meanshift_bandwidth:
            cluster_kwargs = {'bandwidth': meanshift_bandwidth}
        
        folder.perform_clustering(cluster_algorithm, n_clusters, use_transformed, **cluster_kwargs)
    
    for i, sentence in enumerate(folder.sentences):
        is_selected = i in selected_points
        
        # Get cluster assignment
        cluster_idx = folder.clusters[i] if folder.clusters is not None and i < len(folder.clusters) else -1
        cluster_color = CLUSTER_COLORS[cluster_idx % len(CLUSTER_COLORS)] if cluster_idx >= 0 else 'transparent'
        cluster_label = f"C{cluster_idx}" if cluster_idx >= 0 else ""
        
        style = {
            'padding': '12px',
            'margin': '5px',
            'borderRadius': '8px',
            'cursor': 'pointer',
            'backgroundColor': '#f8f9fa' if is_selected else 'white',
            'border': '2px solid #1f2937' if is_selected else '1px solid #dee2e6',
            'borderLeft': f'4px solid {cluster_color}' if cluster_idx >= 0 else 
                        ('4px solid #1f2937' if is_selected else '4px solid transparent'),
            'fontWeight': '600' if is_selected else 'normal',
            'transition': 'all 0.2s ease'
        }
        
        sentence_divs.append(
            html.Div([
                html.Span(f"{i:3d}", style={
                    'color': '#6c757d', 
                    'fontFamily': 'monospace',
                    'marginRight': '10px',
                    'fontSize': '0.875rem'
                }),
                html.Span(sentence, style={'flex': '1'}),
                html.Span(cluster_label, style={
                    'backgroundColor': cluster_color,
                    'color': 'white',
                    'padding': '2px 8px',
                    'borderRadius': '4px',
                    'fontSize': '0.75rem',
                    'fontWeight': '500',
                    'marginLeft': '10px'
                }) if cluster_label else None
            ], 
            id={'type': 'sentence-div', 'index': i},
            style=style,
            n_clicks=0)
        )
    
    # Calculate cluster distribution
    if folder.clusters is not None:
        unique_clusters = set(folder.clusters)
        cluster_counts = {}
        for c in folder.clusters:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
        
        cluster_info = " | ".join([f"C{i}: {cluster_counts.get(i, 0)}" for i in sorted(cluster_counts.keys())])
        stats = f"{len(folder.sentences)} sentences | {len(selected_points)} selected | {cluster_info}"
    else:
       stats = f"{len(folder.sentences)} sentences | {len(selected_points)} selected"
   
    return sentence_divs, stats

# Cluster analysis panel callback
@app.callback(
    Output('cluster-analysis', 'children'),
    [Input('cluster-algorithm-select', 'value'),
     Input('clusters-input', 'value'),
     Input('cluster-on-select', 'value'),
     Input('data-store', 'data')]
)
def update_cluster_analysis(cluster_algorithm, n_clusters, cluster_on, data):
    if folder.clusters is None or not folder.sentences:
        return None
    
    # Count points in each cluster
    unique_clusters = sorted(set(folder.clusters))
    cluster_counts = {}
    for c in folder.clusters:
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
    
    # Create cluster cards
    cluster_cards = []
    for i in unique_clusters:
        count = cluster_counts.get(i, 0)
        percentage = (count / len(folder.clusters) * 100) if len(folder.clusters) > 0 else 0
        
        cluster_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div(style={
                                'width': '16px',
                                'height': '16px',
                                'backgroundColor': CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                                'borderRadius': '50%',
                                'display': 'inline-block',
                                'marginRight': '8px'
                            }),
                            html.Span(f"Cluster {i}", style={'fontWeight': 'bold'})
                        ]),
                        html.Div(f"{count} pts ({percentage:.1f}%)", style={
                            'fontSize': '0.875rem',
                            'color': '#6c757d',
                            'marginTop': '4px'
                        })
                    ], className="p-2")
                ])
            ], width="auto")
        )
    
    algorithm_names = {
        'kmeans': 'K-Means',
        'dbscan': 'DBSCAN',
        'hierarchical': 'Hierarchical',
        'meanshift': 'Mean Shift',
        'spectral': 'Spectral',
        'gmm': 'Gaussian Mixture'
    }
    
    return dbc.Card([
        dbc.CardHeader(f"Cluster Analysis - {algorithm_names.get(cluster_algorithm, cluster_algorithm)} ({len(unique_clusters)} clusters found, {cluster_on} space)"),
        dbc.CardBody([
            dbc.Row(cluster_cards)
        ])
    ])

# Elbow plot callback
@app.callback(
    [Output('elbow-collapse', 'is_open'),
     Output('elbow-plot', 'figure'),
     Output('saved-cluster-assignments', 'data')],
    [Input('elbow-btn', 'n_clicks')],
    [State('elbow-collapse', 'is_open'),
     State('cluster-on-select', 'value')]
)
def toggle_elbow(n_clicks, is_open, cluster_on):
    if n_clicks:
        if not is_open and folder.original_points is not None:
            # Generate elbow plot
            use_transformed = cluster_on == 'transformed'
            k_values, wcss, saved_assignments = folder.compute_elbow_data(10, use_transformed)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=k_values,
                y=wcss,
                mode='lines+markers',
                marker=dict(size=8, color='#8b5cf6'),
                line=dict(width=2, color='#8b5cf6')
            ))
            
            fig.update_layout(
                xaxis_title='Number of Clusters (k)',
                yaxis_title='Within-Cluster Sum of Squares (WCSS)',
                height=300,
                margin=dict(l=50, r=50, t=30, b=50),
                plot_bgcolor='white',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray')
            )
            
            return not is_open, fig, saved_assignments
        else:
            return not is_open, {}, {}
    
    return is_open, {}, {}

if __name__ == '__main__':
    app.run(debug=True, port=8050)
