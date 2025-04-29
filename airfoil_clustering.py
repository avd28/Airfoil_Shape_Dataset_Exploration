import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from typing import List, Tuple
import warnings
from clustering_utils import (
    evaluate_clustering_metrics,
    find_optimal_clusters,
    visualize_optimal_clusters,
    perform_pca_analysis
)
warnings.filterwarnings('ignore')

# Connect to database and load data
@st.cache_data
def load_data():
    conn = sqlite3.connect('airfoil_data.db')
    df = pd.read_sql_query("""
        SELECT airfoilName, coefficientLift, coefficientDrag, coefficientMoment, 
               reynoldsNumber, alpha 
        FROM airfoils
    """, conn)
    conn.close()
    return df

# Preprocess data for clustering
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Group by airfoil and calculate mean values for each parameter
    airfoil_stats = df.groupby('airfoilName').agg({
        'coefficientLift': ['mean', 'max', 'min', 'std'],
        'coefficientDrag': ['mean', 'min', 'std'],
        'coefficientMoment': ['mean', 'std'],
        'reynoldsNumber': ['mean'],
        'alpha': ['mean', 'std']
    }).reset_index()
    
    # Flatten multi-index columns
    airfoil_stats.columns = ['_'.join(col).strip('_') for col in airfoil_stats.columns.values]
    
    # Select features for clustering
    features = airfoil_stats.drop('airfoilName', axis=1)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )
    
    return airfoil_stats, scaled_features

# Perform clustering with different algorithms
def perform_clustering(scaled_features: pd.DataFrame, n_clusters: int = 5) -> dict:
    results = {}
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    results['kmeans'] = kmeans.fit_predict(scaled_features)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    results['dbscan'] = dbscan.fit_predict(scaled_features)
    
    # Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    results['agg'] = agg.fit_predict(scaled_features)
    
    return results

# Visualize clusters
def visualize_clusters(airfoil_stats: pd.DataFrame, results: dict, method: str):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Lift vs Drag', 'Lift vs Moment',
                                      'Drag vs Moment', 'Reynolds vs Alpha'))
    
    # Add traces for each cluster
    for cluster in np.unique(results[method]):
        mask = results[method] == cluster
        cluster_data = airfoil_stats[mask]
        
        # Lift vs Drag
        fig.add_trace(
            go.Scatter(
                x=cluster_data['coefficientDrag_mean'],
                y=cluster_data['coefficientLift_mean'],
                mode='markers',
                name=f'Cluster {cluster}',
                text=cluster_data['airfoilName'],
                hovertemplate='%{text}<br>Drag: %{x:.3f}<br>Lift: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Lift vs Moment
        fig.add_trace(
            go.Scatter(
                x=cluster_data['coefficientMoment_mean'],
                y=cluster_data['coefficientLift_mean'],
                mode='markers',
                name=f'Cluster {cluster}',
                text=cluster_data['airfoilName'],
                hovertemplate='%{text}<br>Moment: %{x:.3f}<br>Lift: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Drag vs Moment
        fig.add_trace(
            go.Scatter(
                x=cluster_data['coefficientMoment_mean'],
                y=cluster_data['coefficientDrag_mean'],
                mode='markers',
                name=f'Cluster {cluster}',
                text=cluster_data['airfoilName'],
                hovertemplate='%{text}<br>Moment: %{x:.3f}<br>Drag: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Reynolds vs Alpha
        fig.add_trace(
            go.Scatter(
                x=cluster_data['reynoldsNumber_mean'],
                y=cluster_data['alpha_mean'],
                mode='markers',
                name=f'Cluster {cluster}',
                text=cluster_data['airfoilName'],
                hovertemplate='%{text}<br>Re: %{x:.0f}<br>Alpha: %{y:.1f}<extra></extra>'
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=1000, width=1200, title_text=f"Airfoil Clusters - {method}")
    return fig

# Visualize PCA results
def visualize_pca(pca_data: np.ndarray, labels: np.ndarray, airfoil_names: List[str]) -> go.Figure:
    fig = go.Figure()
    
    for cluster in np.unique(labels):
        mask = labels == cluster
        fig.add_trace(
            go.Scatter(
                x=pca_data[mask, 0],
                y=pca_data[mask, 1],
                mode='markers',
                name=f'Cluster {cluster}',
                text=airfoil_names[mask],
                hovertemplate='%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
            )
        )
    
    fig.update_layout(
        title="PCA Visualization of Airfoil Clusters",
        xaxis_title="First Principal Component",
        yaxis_title="Second Principal Component",
        height=600,
        width=800
    )
    return fig

# Streamlit app
def main():
    st.title("Airfoil Similarity Analysis")
    
    # Load and preprocess data
    df = load_data()
    airfoil_stats, scaled_features = preprocess_data(df)
    
    # Sidebar controls
    st.sidebar.header("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
    method = st.sidebar.selectbox("Clustering Method", 
                                ["kmeans", "dbscan", "agg"])
    
    # Find optimal number of clusters
    st.header("Optimal Number of Clusters")
    optimal_n, scores = find_optimal_clusters(scaled_features)
    st.write(f"Optimal number of clusters: {optimal_n}")
    
    # Visualize clustering metrics
    metrics_fig = visualize_optimal_clusters(scores)
    st.plotly_chart(metrics_fig, use_container_width=True)
    
    # Perform clustering
    results = perform_clustering(scaled_features, n_clusters)
    
    # Evaluate clustering metrics
    metrics = evaluate_clustering_metrics(scaled_features, results[method])
    st.header("Clustering Results")
    st.write("Clustering Metrics:")
    st.write(f"- Silhouette Score: {metrics['silhouette_score']:.3f}")
    st.write(f"- Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.3f}")
    st.write(f"- Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
    
    # Visualize clusters
    fig = visualize_clusters(airfoil_stats, results, method)
    st.plotly_chart(fig, use_container_width=True)
    
    # Perform and visualize PCA
    st.header("PCA Analysis")
    pca_data, explained_variance = perform_pca_analysis(scaled_features)
    st.write(f"Explained Variance Ratio: {explained_variance[0]:.3f}, {explained_variance[1]:.3f}")
    
    pca_fig = visualize_pca(pca_data, results[method], airfoil_stats['airfoilName'].values)
    st.plotly_chart(pca_fig, use_container_width=True)
    
    # Display cluster members
    st.header("Cluster Members")
    for cluster in np.unique(results[method]):
        cluster_airfoils = airfoil_stats[results[method] == cluster]['airfoilName']
        st.subheader(f"Cluster {cluster}")
        st.write(cluster_airfoils.tolist())

if __name__ == "__main__":
    main() 