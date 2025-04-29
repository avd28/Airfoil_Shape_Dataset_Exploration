import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Dict

def evaluate_clustering_metrics(scaled_features: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate various clustering metrics to evaluate the quality of clusters.
    
    Args:
        scaled_features: Scaled feature matrix
        labels: Cluster labels
        
    Returns:
        Dictionary containing different clustering metrics
    """
    metrics = {}
    
    if len(np.unique(labels)) > 1:
        metrics['silhouette_score'] = silhouette_score(scaled_features, labels)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(scaled_features, labels)
        metrics['davies_bouldin_score'] = davies_bouldin_score(scaled_features, labels)
    else:
        metrics['silhouette_score'] = None
        metrics['calinski_harabasz_score'] = None
        metrics['davies_bouldin_score'] = None
        
    return metrics

def find_optimal_clusters(scaled_features: pd.DataFrame, 
                         min_clusters: int = 2, 
                         max_clusters: int = 10) -> Tuple[int, Dict[str, List[float]]]:
    """
    Find the optimal number of clusters using the elbow method and silhouette scores.
    
    Args:
        scaled_features: Scaled feature matrix
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        
    Returns:
        Tuple containing the optimal number of clusters and a dictionary of scores
    """
    from sklearn.cluster import KMeans
    
    scores = {
        'inertia': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled_features)
        
        scores['inertia'].append(kmeans.inertia_)
        if len(np.unique(labels)) > 1:
            scores['silhouette'].append(silhouette_score(scaled_features, labels))
            scores['calinski_harabasz'].append(calinski_harabasz_score(scaled_features, labels))
            scores['davies_bouldin'].append(davies_bouldin_score(scaled_features, labels))
        else:
            scores['silhouette'].append(None)
            scores['calinski_harabasz'].append(None)
            scores['davies_bouldin'].append(None)
    
    # Find optimal number of clusters using silhouette score
    valid_scores = [s for s in scores['silhouette'] if s is not None]
    if valid_scores:
        optimal_clusters = np.argmax(valid_scores) + min_clusters
    else:
        optimal_clusters = min_clusters
    
    return optimal_clusters, scores

def visualize_optimal_clusters(scores: Dict[str, List[float]], 
                             min_clusters: int = 2, 
                             max_clusters: int = 10) -> go.Figure:
    """
    Create visualization of clustering metrics across different numbers of clusters.
    
    Args:
        scores: Dictionary containing clustering metrics
        min_clusters: Minimum number of clusters
        max_clusters: Maximum number of clusters
        
    Returns:
        Plotly figure showing the metrics
    """
    n_clusters = list(range(min_clusters, max_clusters + 1))
    
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Inertia', 'Silhouette Score',
                                      'Calinski-Harabasz Score', 'Davies-Bouldin Score'))
    
    # Inertia
    fig.add_trace(
        go.Scatter(x=n_clusters, y=scores['inertia'], mode='lines+markers', name='Inertia'),
        row=1, col=1
    )
    
    # Silhouette Score
    fig.add_trace(
        go.Scatter(x=n_clusters, y=scores['silhouette'], mode='lines+markers', name='Silhouette'),
        row=1, col=2
    )
    
    # Calinski-Harabasz Score
    fig.add_trace(
        go.Scatter(x=n_clusters, y=scores['calinski_harabasz'], mode='lines+markers', name='Calinski-Harabasz'),
        row=2, col=1
    )
    
    # Davies-Bouldin Score
    fig.add_trace(
        go.Scatter(x=n_clusters, y=scores['davies_bouldin'], mode='lines+markers', name='Davies-Bouldin'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, width=1200, title_text="Clustering Metrics")
    return fig

def perform_pca_analysis(scaled_features: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform PCA analysis on the scaled features.
    
    Args:
        scaled_features: Scaled feature matrix
        n_components: Number of principal components to keep
        
    Returns:
        Tuple containing the transformed data and explained variance ratio
    """
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(scaled_features)
    explained_variance = pca.explained_variance_ratio_
    
    return transformed_data, explained_variance 