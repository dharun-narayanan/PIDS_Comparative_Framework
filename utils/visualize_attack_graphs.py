#!/usr/bin/env python3
"""
Attack Graph Visualization and Interactive Comparison

This script creates meaningful attack summary graphs from anomaly scores,
reconstructs attack paths through provenance traversal, and provides
interactive multi-model visualization.

Features:
- Robust attack graph reconstruction from scores
- Backward/forward provenance traversal
- Multi-model interactive comparison
- Entity clustering and path ranking
- Export to multiple formats (HTML, JSON, GraphML)
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime
import logging
from scipy.sparse import csr_matrix
import webbrowser
import platform
import subprocess
import socket
import threading
import http.server
import socketserver

# Visualization imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Install with: pip install plotly kaleido")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.common import setup_logging

logger = setup_logging()
logger = logging.getLogger("visualize_attack_graphs")


def open_in_browser(filepath: Path) -> bool:
    """
    Open HTML file in default browser with cross-platform support.
    
    Args:
        filepath: Path to HTML file to open
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_url = f'file://{filepath.absolute()}'
        
        # Try platform-specific approaches first for better reliability
        system = platform.system()
        
        if system == 'Linux':
            # Try xdg-open first (most common on Linux)
            try:
                subprocess.run(['xdg-open', str(filepath)], check=True, 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        elif system == 'Darwin':  # macOS
            try:
                subprocess.run(['open', str(filepath)], check=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        elif system == 'Windows':
            try:
                os.startfile(str(filepath))
                return True
            except Exception:
                pass
        
        # Fallback to webbrowser module (works on all platforms)
        webbrowser.open(file_url)
        return True
        
    except Exception as e:
        logger.warning(f"Could not auto-open browser: {e}")
        return False


def is_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
            return True
    except OSError:
        return False


def start_http_server(directory: Path, port: int = 8000) -> Optional[int]:
    """
    Start a simple HTTP server in a background thread.
    
    Args:
        directory: Directory to serve files from
        port: Port to start the server on (will try next ports if occupied)
        
    Returns:
        Port number if successful, None otherwise
    """
    # Find an available port
    max_attempts = 10
    current_port = port
    
    for _ in range(max_attempts):
        if is_port_available(current_port):
            break
        current_port += 1
    else:
        logger.error(f"Could not find available port in range {port}-{current_port}")
        return None
    
    try:
        # Create a custom handler that serves from the specified directory
        # and redirects root to the HTML file
        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(directory), **kwargs)
            
            def do_GET(self):
                # Redirect root path to the HTML file
                if self.path == '/' or self.path == '':
                    self.send_response(302)
                    self.send_header('Location', '/attack_graph_viewer.html')
                    self.end_headers()
                    return
                # Serve other files normally
                super().do_GET()
        
        CustomHandler.extensions_map['.html'] = 'text/html'
        CustomHandler.extensions_map['.json'] = 'application/json'
        
        httpd = socketserver.TCPServer(("", current_port), CustomHandler)
        
        # Start server in background thread
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        
        logger.info(f"✓ HTTP server started on port {current_port}")
        return current_port
        
    except Exception as e:
        logger.error(f"Failed to start HTTP server: {e}")
        return None


class AttackGraphReconstructor:
    """Reconstructs meaningful attack graphs from anomaly scores."""
    
    def __init__(
        self,
        graph_data: Dict,
        anomaly_scores: np.ndarray,
        threshold_percentile: float = 95.0,
        max_path_length: int = 10,
        min_path_score: float = 0.0
    ):
        """
        Initialize attack graph reconstructor.
        
        Args:
            graph_data: Loaded provenance graph with edges, node mappings
            anomaly_scores: Anomaly scores for each edge
            threshold_percentile: Percentile threshold for anomalies
            max_path_length: Maximum path length for traversal
            min_path_score: Minimum average score for paths
        """
        self.graph_data = graph_data
        self.anomaly_scores = anomaly_scores
        self.threshold_percentile = threshold_percentile
        self.max_path_length = max_path_length
        self.min_path_score = min_path_score
        
        # Extract graph from the actual data structure
        self.G = self._extract_graph()
        logger.info(f"Extracted graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # Compute threshold
        self.threshold = np.percentile(anomaly_scores, threshold_percentile)
        logger.info(f"Anomaly threshold ({threshold_percentile}%): {self.threshold:.6f}")
        
        # Identify anomalous edges
        self.anomalous_edges = self._identify_anomalous_edges()
        logger.info(f"Found {len(self.anomalous_edges)} anomalous edges")
    
    def _extract_graph(self) -> nx.DiGraph:
        """Extract NetworkX graph from graph data."""
        G = nx.DiGraph()
        
        # Handle different data structures
        if isinstance(self.graph_data, nx.DiGraph):
            return self.graph_data
        
        if not isinstance(self.graph_data, dict):
            logger.error(f"Unsupported graph data type: {type(self.graph_data)}")
            return G
        
        # Check if 'graph' key exists (for backward compatibility)
        if 'graph' in self.graph_data:
            return self.graph_data['graph']
        
        # Extract from the actual data structure: edges, node_id_map, etc.
        edges = self.graph_data.get('edges', [])
        node_id_map = self.graph_data.get('node_id_map', {})
        node_type_map = self.graph_data.get('node_type_map', {})
        edge_type_map = self.graph_data.get('edge_type_map', {})
        timestamps = self.graph_data.get('timestamps', [])
        node_features = self.graph_data.get('node_features', None)
        
        logger.info(f"Building graph from {len(edges)} edges and {len(node_id_map)} nodes")
        
        # Create reverse mapping for node IDs
        id_to_node = {v: k for k, v in node_id_map.items()}
        
        # Add nodes with attributes
        for node_id, node_idx in node_id_map.items():
            node_type = node_type_map.get(node_idx, 'unknown')
            attrs = {
                'node_id': node_idx,
                'node_type': node_type,
                'name': str(node_id)[:50]  # Truncate long names
            }
            
            # Add node features if available (check if it's a valid array index)
            if node_features is not None and isinstance(node_features, (list, np.ndarray)):
                if node_idx < len(node_features):
                    attrs['features'] = node_features[node_idx]
            
            G.add_node(node_idx, **attrs)
        
        # Add edges with attributes
        edge_count = 0
        for edge_idx, edge in enumerate(edges):
            if edge_idx >= len(self.anomaly_scores):
                break
                
            # Handle different edge formats
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                src, dst = edge[0], edge[1]
            elif isinstance(edge, dict):
                src = edge.get('source', edge.get('src', edge.get(0)))
                dst = edge.get('target', edge.get('dst', edge.get(1)))
            else:
                logger.warning(f"Unknown edge format at index {edge_idx}: {type(edge)}")
                continue
            
            # Get edge attributes
            edge_type = edge_type_map.get(edge_idx, 'unknown')
            timestamp = timestamps[edge_idx] if edge_idx < len(timestamps) else 0
            
            attrs = {
                'edge_id': edge_idx,
                'event_type': edge_type,
                'timestamp': timestamp,
                'anomaly_score': float(self.anomaly_scores[edge_idx])
            }
            
            # Add edge to graph
            if src in G and dst in G:
                G.add_edge(src, dst, **attrs)
                edge_count += 1
            else:
                # Try to add nodes if missing
                if src not in G:
                    G.add_node(src, node_id=src, node_type='unknown', name=f'node_{src}')
                if dst not in G:
                    G.add_node(dst, node_id=dst, node_type='unknown', name=f'node_{dst}')
                G.add_edge(src, dst, **attrs)
                edge_count += 1
        
        logger.info(f"Created graph with {G.number_of_nodes()} nodes and {edge_count} edges")
        return G

    
    def _identify_anomalous_edges(self) -> List[Tuple[int, int, Dict]]:
        """Identify edges with anomaly scores above threshold."""
        anomalous = []
        
        for u, v, attrs in self.G.edges(data=True):
            edge_id = attrs.get('edge_id')
            if edge_id is not None and edge_id < len(self.anomaly_scores):
                score = self.anomaly_scores[edge_id]
                if score >= self.threshold:
                    edge_attrs = dict(attrs)
                    edge_attrs['anomaly_score'] = float(score)
                    anomalous.append((u, v, edge_attrs))
        
        return sorted(anomalous, key=lambda x: x[2]['anomaly_score'], reverse=True)
    
    def _get_edge_by_index(self, edge_id: int) -> Optional[Tuple[Any, Any, Dict]]:
        """Get edge by its index."""
        for u, v, attrs in self.G.edges(data=True):
            if attrs.get('edge_id') == edge_id:
                return (u, v, attrs)
        return None
    
    def reconstruct_attack_graph(
        self,
        top_k: int = 100,
        cluster_by: str = 'entity'
    ) -> nx.DiGraph:
        """
        Reconstruct attack graph from anomalous edges.
        
        Args:
            top_k: Number of top anomalies to include
            cluster_by: Clustering strategy ('entity', 'temporal', 'path')
            
        Returns:
            Reconstructed attack graph
        """
        logger.info(f"Reconstructing attack graph (top_k={top_k}, cluster={cluster_by})")
        
        # Create attack graph
        attack_graph = nx.DiGraph()
        
        # Process top-k anomalies
        top_anomalies = self.anomalous_edges[:top_k]
        
        for src, dst, attrs in top_anomalies:
            # Add nodes with attributes
            if src not in attack_graph:
                src_attrs = self.G.nodes.get(src, {})
                attack_graph.add_node(src, **src_attrs)
            
            if dst not in attack_graph:
                dst_attrs = self.G.nodes.get(dst, {})
                attack_graph.add_node(dst, **dst_attrs)
            
            # Add edge
            attack_graph.add_edge(src, dst, **attrs)
        
        # Expand with provenance context
        attack_graph = self._expand_with_context(attack_graph, top_anomalies)
        
        # Apply clustering
        if cluster_by == 'entity':
            attack_graph = self._cluster_by_entity(attack_graph)
        elif cluster_by == 'temporal':
            attack_graph = self._cluster_by_time(attack_graph)
        elif cluster_by == 'path':
            attack_graph = self._cluster_by_path(attack_graph)
        
        # Compute graph metrics
        attack_graph = self._compute_graph_metrics(attack_graph)
        
        logger.info(f"Attack graph: {attack_graph.number_of_nodes()} nodes, "
                   f"{attack_graph.number_of_edges()} edges")
        
        return attack_graph
    
    def _expand_with_context(
        self,
        attack_graph: nx.DiGraph,
        anomalies: List[Tuple]
    ) -> nx.DiGraph:
        """Expand attack graph with provenance context (optimized)."""
        logger.info("Expanding with provenance context...")
        
        # Collect seed nodes (high-anomaly entities)
        seed_nodes = set()
        for src, dst, attrs in anomalies[:50]:  # Limit to top 50 to avoid long running time
            seed_nodes.add(src)
            seed_nodes.add(dst)
        
        logger.info(f"Expanding from {len(seed_nodes)} seed nodes")
        
        # Track visited nodes to avoid redundant work
        visited_backward = set()
        visited_forward = set()
        
        # Backward provenance (find attack origins) - limited depth
        for node in list(seed_nodes):
            if node not in visited_backward:
                self._backward_traversal_optimized(
                    node, attack_graph, visited_backward, max_depth=2
                )
        
        # Forward provenance (find attack consequences) - limited depth
        for node in list(seed_nodes):
            if node not in visited_forward:
                self._forward_traversal_optimized(
                    node, attack_graph, visited_forward, max_depth=2
                )
        
        logger.info(f"Expanded graph: {attack_graph.number_of_nodes()} nodes, "
                   f"{attack_graph.number_of_edges()} edges")
        
        return attack_graph
    
    def _backward_traversal_optimized(
        self,
        node: Any,
        attack_graph: nx.DiGraph,
        visited: Set,
        max_depth: int = 2,
        current_depth: int = 0
    ):
        """Optimized backward traversal to find attack origins."""
        if current_depth >= max_depth or node in visited:
            return
        
        visited.add(node)
        
        # Get predecessors
        predecessors = list(self.G.predecessors(node))
        
        # Limit number of predecessors to explore
        if len(predecessors) > 10:
            # Prioritize by anomaly score
            pred_scores = []
            for pred in predecessors:
                edge_attrs = self.G.get_edge_data(pred, node, {})
                score = edge_attrs.get('anomaly_score', 0.0)
                pred_scores.append((pred, score))
            pred_scores.sort(key=lambda x: x[1], reverse=True)
            predecessors = [p for p, s in pred_scores[:10]]
        
        for pred in predecessors:
            edge_attrs = self.G.get_edge_data(pred, node, {})
            score = edge_attrs.get('anomaly_score', 0.0)
            
            # Add if score is relevant or connects to anomalous node
            if score > self.threshold * 0.3 or pred in attack_graph:
                if pred not in attack_graph:
                    pred_attrs = self.G.nodes.get(pred, {})
                    attack_graph.add_node(pred, **pred_attrs, context='backward')
                
                edge_attrs_copy = dict(edge_attrs)
                edge_attrs_copy['provenance_type'] = 'backward'
                attack_graph.add_edge(pred, node, **edge_attrs_copy)
                
                # Continue traversal
                self._backward_traversal_optimized(
                    pred, attack_graph, visited, max_depth, current_depth + 1
                )
    
    def _forward_traversal_optimized(
        self,
        node: Any,
        attack_graph: nx.DiGraph,
        visited: Set,
        max_depth: int = 2,
        current_depth: int = 0
    ):
        """Optimized forward traversal to find attack consequences."""
        if current_depth >= max_depth or node in visited:
            return
        
        visited.add(node)
        
        # Get successors
        successors = list(self.G.successors(node))
        
        # Limit number of successors to explore
        if len(successors) > 10:
            # Prioritize by anomaly score
            succ_scores = []
            for succ in successors:
                edge_attrs = self.G.get_edge_data(node, succ, {})
                score = edge_attrs.get('anomaly_score', 0.0)
                succ_scores.append((succ, score))
            succ_scores.sort(key=lambda x: x[1], reverse=True)
            successors = [s for s, sc in succ_scores[:10]]
        
        for succ in successors:
            edge_attrs = self.G.get_edge_data(node, succ, {})
            score = edge_attrs.get('anomaly_score', 0.0)
            
            # Add if score is relevant or connects to anomalous node
            if score > self.threshold * 0.3 or succ in attack_graph:
                if succ not in attack_graph:
                    succ_attrs = self.G.nodes.get(succ, {})
                    attack_graph.add_node(succ, **succ_attrs, context='forward')
                
                edge_attrs_copy = dict(edge_attrs)
                edge_attrs_copy['provenance_type'] = 'forward'
                attack_graph.add_edge(node, succ, **edge_attrs_copy)
                
                # Continue traversal
                self._forward_traversal_optimized(
                    succ, attack_graph, visited, max_depth, current_depth + 1
                )
    
    def _cluster_by_entity(self, attack_graph: nx.DiGraph) -> nx.DiGraph:
        """Cluster nodes by entity type."""
        for node, attrs in attack_graph.nodes(data=True):
            node_type = attrs.get('node_type', attrs.get('type', 'unknown'))
            attrs['cluster'] = node_type
        return attack_graph
    
    def _cluster_by_time(self, attack_graph: nx.DiGraph) -> nx.DiGraph:
        """Cluster nodes by temporal windows."""
        # Extract timestamps
        timestamps = []
        for u, v, attrs in attack_graph.edges(data=True):
            ts = attrs.get('timestamp', 0)
            if ts > 0:
                timestamps.append(ts)
        
        if not timestamps:
            return attack_graph
        
        # Create time bins
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        duration = max_ts - min_ts
        bin_size = duration / 10  # 10 time windows
        
        # Assign clusters
        for node, attrs in attack_graph.nodes(data=True):
            # Find representative timestamp for node
            node_timestamps = []
            for u, v, edge_attrs in attack_graph.edges(node, data=True):
                ts = edge_attrs.get('timestamp', 0)
                if ts > 0:
                    node_timestamps.append(ts)
            
            if node_timestamps:
                avg_ts = np.mean(node_timestamps)
                cluster = int((avg_ts - min_ts) / bin_size) if bin_size > 0 else 0
                attrs['cluster'] = f"time_{cluster}"
            else:
                attrs['cluster'] = "time_unknown"
        
        return attack_graph
    
    def _cluster_by_path(self, attack_graph: nx.DiGraph) -> nx.DiGraph:
        """Cluster nodes by attack paths."""
        # Find connected components
        undirected = attack_graph.to_undirected()
        components = list(nx.connected_components(undirected))
        
        # Assign cluster IDs
        for i, component in enumerate(components):
            for node in component:
                attack_graph.nodes[node]['cluster'] = f"path_{i}"
        
        return attack_graph
    
    def _compute_graph_metrics(self, attack_graph: nx.DiGraph) -> nx.DiGraph:
        """Compute graph-level metrics for nodes."""
        logger.info("Computing graph metrics...")
        
        # Degree centrality
        in_degree = dict(attack_graph.in_degree())
        out_degree = dict(attack_graph.out_degree())
        
        # Betweenness centrality (on undirected version for speed)
        try:
            betweenness = nx.betweenness_centrality(attack_graph.to_undirected())
        except:
            betweenness = {node: 0.0 for node in attack_graph.nodes()}
        
        # PageRank (anomaly propagation)
        try:
            pagerank = nx.pagerank(attack_graph)
        except:
            pagerank = {node: 1.0 / attack_graph.number_of_nodes() 
                       for node in attack_graph.nodes()}
        
        # Add metrics to nodes
        for node in attack_graph.nodes():
            attack_graph.nodes[node]['in_degree'] = in_degree.get(node, 0)
            attack_graph.nodes[node]['out_degree'] = out_degree.get(node, 0)
            attack_graph.nodes[node]['betweenness'] = betweenness.get(node, 0.0)
            attack_graph.nodes[node]['pagerank'] = pagerank.get(node, 0.0)
            attack_graph.nodes[node]['importance'] = (
                betweenness.get(node, 0.0) * 0.5 + pagerank.get(node, 0.0) * 0.5
            )
        
        return attack_graph
    
    def extract_attack_paths(
        self,
        attack_graph: nx.DiGraph,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Extract top-k attack paths from attack graph (optimized).
        
        Returns:
            List of attack paths with scores and details
        """
        logger.info(f"Extracting top-{top_k} attack paths...")
        
        paths = []
        
        # Find all simple paths between high-anomaly nodes (optimized)
        high_anomaly_nodes = sorted(
            attack_graph.nodes(),
            key=lambda n: attack_graph.nodes[n].get('importance', 0),
            reverse=True
        )[:15]  # Limit to top 15 nodes to avoid long running time
        
        # Limit path search space
        max_paths_per_pair = 3
        
        for i, src in enumerate(high_anomaly_nodes):
            for dst in high_anomaly_nodes[i+1:]:  # Avoid duplicate pairs
                if src == dst:
                    continue
                
                try:
                    # Find limited number of simple paths
                    path_count = 0
                    for path in nx.all_simple_paths(
                        attack_graph, src, dst, cutoff=min(self.max_path_length, 6)
                    ):
                        if len(path) >= 2:
                            # Compute path score
                            path_score = self._compute_path_score(attack_graph, path)
                            
                            if path_score >= self.min_path_score:
                                paths.append({
                                    'path': path,
                                    'length': len(path),
                                    'score': path_score,
                                    'nodes': [self._get_node_info(attack_graph, n) for n in path],
                                    'edges': self._get_path_edges(attack_graph, path)
                                })
                                
                                path_count += 1
                                if path_count >= max_paths_per_pair:
                                    break
                
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    logger.debug(f"Error finding paths from {src} to {dst}: {e}")
                    continue
        
        # Sort by score and return top-k
        paths = sorted(paths, key=lambda x: x['score'], reverse=True)[:top_k]
        
        logger.info(f"Extracted {len(paths)} attack paths")
        return paths
    
    def _compute_path_score(self, G: nx.DiGraph, path: List) -> float:
        """Compute anomaly score for a path."""
        scores = []
        for i in range(len(path) - 1):
            edge_attrs = G.get_edge_data(path[i], path[i+1], {})
            score = edge_attrs.get('anomaly_score', 0.0)
            scores.append(score)
        
        if not scores:
            return 0.0
        
        # Use max score (most anomalous edge) with length penalty
        max_score = max(scores)
        avg_score = np.mean(scores)
        
        # Combine max and average, with preference for shorter paths
        combined_score = (max_score * 0.7 + avg_score * 0.3) / np.log2(len(path) + 1)
        
        return combined_score
    
    def _get_node_info(self, G: nx.DiGraph, node: Any) -> Dict:
        """Get node information."""
        attrs = G.nodes.get(node, {})
        return {
            'id': str(node),
            'type': attrs.get('node_type', attrs.get('type', 'unknown')),
            'name': attrs.get('name', attrs.get('label', str(node))),
            'importance': attrs.get('importance', 0.0)
        }
    
    def _get_path_edges(self, G: nx.DiGraph, path: List) -> List[Dict]:
        """Get edge information for a path."""
        edges = []
        for i in range(len(path) - 1):
            attrs = G.get_edge_data(path[i], path[i+1], {})
            edges.append({
                'source': str(path[i]),
                'target': str(path[i+1]),
                'type': attrs.get('event_type', attrs.get('type', 'unknown')),
                'score': attrs.get('anomaly_score', 0.0),
                'timestamp': attrs.get('timestamp', 0)
            })
        return edges


class MultiModelVisualizer:
    """Interactive multi-model attack graph visualizer with navigation."""
    
    def __init__(self, output_dir: Path):
        """Initialize visualizer."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Install with: pip install plotly kaleido")
    
    def create_interactive_comparison(
        self,
        model_graphs: Dict[str, nx.DiGraph],
        model_paths: Dict[str, List[Dict]],
        model_scores: Dict[str, np.ndarray]
    ) -> str:
        """
        Create interactive single-graph view with model navigation.
        
        Args:
            model_graphs: Dict of model_name -> attack_graph
            model_paths: Dict of model_name -> attack_paths
            model_scores: Dict of model_name -> anomaly_scores
            
        Returns:
            Path to HTML file
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly required for interactive visualization")
            return None
        
        logger.info("Creating interactive visualization with model navigation...")
        
        # Create HTML with tabs for each model
        html_content = self._create_html_template(model_graphs, model_paths, model_scores)
        
        # Save HTML
        html_path = self.output_dir / "attack_graph_viewer.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Saved interactive visualization: {html_path}")
        return str(html_path)
    
    def _create_html_template(
        self,
        model_graphs: Dict[str, nx.DiGraph],
        model_paths: Dict[str, List[Dict]],
        model_scores: Dict[str, np.ndarray]
    ) -> str:
        """Create HTML template with navigation and graphs."""
        
        # Generate graph data for each model
        graphs_data = {}
        stats_data = {}
        
        for model_name in model_graphs.keys():
            G = model_graphs[model_name]
            scores = model_scores[model_name]
            
            # Generate Plotly figure for this model
            fig = self._create_single_graph_figure(G, model_name)
            
            # Convert to JSON
            graphs_data[model_name] = fig.to_json()
            
            # Calculate statistics
            stats_data[model_name] = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'components': nx.number_weakly_connected_components(G),
                'density': nx.density(G),
                'mean_score': float(scores.mean()),
                'max_score': float(scores.max()),
                'min_score': float(scores.min()),
                'p95': float(np.percentile(scores, 95)),
                'p99': float(np.percentile(scores, 99)),
                'total_scores': len(scores)
            }
        
        # Create HTML with navigation
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attack Graph Visualization - Multi-Model Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            color: #2d3748;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #718096;
            font-size: 1.1em;
        }}
        
        .nav-container {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .model-tab {{
            background: white;
            border: none;
            border-radius: 10px;
            padding: 15px 30px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            color: #4a5568;
        }}
        
        .model-tab:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }}
        
        .model-tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: scale(1.05);
        }}
        
        .content-area {{
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
        }}
        
        .graph-container {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            min-height: 700px;
        }}
        
        .stats-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            height: fit-content;
            position: sticky;
            top: 20px;
        }}
        
        .stats-card h3 {{
            font-size: 1.5em;
            color: #2d3748;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
        }}
        
        .stat-item {{
            margin-bottom: 15px;
            padding: 10px;
            background: #f7fafc;
            border-radius: 8px;
        }}
        
        .stat-label {{
            font-size: 0.85em;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 1.3em;
            font-weight: 700;
            color: #2d3748;
        }}
        
        .stat-value.highlight {{
            color: #667eea;
        }}
        
        .graph-view {{
            display: none;
        }}
        
        .graph-view.active {{
            display: block;
        }}
        
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background: #f7fafc;
            border-radius: 10px;
        }}
        
        .legend h4 {{
            font-size: 1em;
            color: #2d3748;
            margin-bottom: 10px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.9em;
        }}
        
        .legend-color {{
            width: 30px;
            height: 15px;
            border-radius: 3px;
            margin-right: 10px;
        }}
        
        @media (max-width: 1200px) {{
            .content-area {{
                grid-template-columns: 1fr;
            }}
            
            .stats-card {{
                position: relative;
                top: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Attack Graph Visualization</h1>
            <p>Interactive provenance-based attack graph analysis across multiple detection models</p>
        </div>
        
        <div class="nav-container" id="modelNav">
            {self._generate_nav_buttons(model_graphs.keys())}
        </div>
        
        <div class="content-area">
            <div class="graph-container">
                {self._generate_graph_views(graphs_data)}
            </div>
            
            <div class="stats-card" id="statsCard">
                {self._generate_stats_cards(stats_data)}
            </div>
        </div>
    </div>
    
    <script>
        // Graph data
        const graphsData = {json.dumps({k: v for k, v in graphs_data.items()})};
        const statsData = {json.dumps(stats_data)};
        
        // Initialize
        let currentModel = '{list(model_graphs.keys())[0]}';
        
        function showModel(modelName) {{
            // Update tabs
            document.querySelectorAll('.model-tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelector(`[data-model="${{modelName}}"]`).classList.add('active');
            
            // Update graph
            document.querySelectorAll('.graph-view').forEach(view => {{
                view.classList.remove('active');
            }});
            document.getElementById(`graph-${{modelName}}`).classList.add('active');
            
            // Update stats
            updateStats(modelName);
            
            // Render graph if not already rendered
            if (!document.getElementById(`graph-${{modelName}}`).hasAttribute('data-rendered')) {{
                const graphData = JSON.parse(graphsData[modelName]);
                Plotly.newPlot(`graph-${{modelName}}`, graphData.data, graphData.layout, {{responsive: true}});
                document.getElementById(`graph-${{modelName}}`).setAttribute('data-rendered', 'true');
            }}
            
            currentModel = modelName;
        }}
        
        function updateStats(modelName) {{
            const stats = statsData[modelName];
            document.getElementById('statsCard').innerHTML = `
                <h3>${{modelName.toUpperCase()}}</h3>
                <div class="stat-item">
                    <div class="stat-label">Total Nodes</div>
                    <div class="stat-value">${{stats.nodes.toLocaleString()}}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Total Edges</div>
                    <div class="stat-value">${{stats.edges.toLocaleString()}}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Connected Components</div>
                    <div class="stat-value">${{stats.components}}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Graph Density</div>
                    <div class="stat-value">${{stats.density.toFixed(4)}}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Anomaly Score Range</div>
                    <div class="stat-value highlight">${{stats.min_score.toFixed(6)}} - ${{stats.max_score.toFixed(6)}}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Mean Score</div>
                    <div class="stat-value">${{stats.mean_score.toFixed(6)}}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">95th Percentile</div>
                    <div class="stat-value highlight">${{stats.p95.toFixed(6)}}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">99th Percentile</div>
                    <div class="stat-value highlight">${{stats.p99.toFixed(6)}}</div>
                </div>
                <div class="legend">
                    <h4>Legend</h4>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FF6B6B;"></div>
                        <span>High Anomaly (>0.8)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FFA500;"></div>
                        <span>Medium Anomaly (0.5-0.8)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #D3D3D3;"></div>
                        <span>Low Anomaly (<0.5)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FF6B6B; border-radius: 50%; width: 15px; height: 15px;"></div>
                        <span>Process Node</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #4ECDC4; border-radius: 50%; width: 15px; height: 15px;"></div>
                        <span>File Node</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #95E1D3; border-radius: 50%; width: 15px; height: 15px;"></div>
                        <span>Network Node</span>
                    </div>
                </div>
            `;
        }}
        
        // Initialize first model
        showModel(currentModel);
    </script>
</body>
</html>
"""
        
        return html
    
    def _generate_nav_buttons(self, model_names) -> str:
        """Generate navigation buttons HTML."""
        buttons = []
        for i, model in enumerate(model_names):
            active = 'active' if i == 0 else ''
            buttons.append(
                f'<button class="model-tab {active}" data-model="{model}" '
                f'onclick="showModel(\'{model}\')">{model.upper()}</button>'
            )
        return '\n'.join(buttons)
    
    def _generate_graph_views(self, graphs_data: Dict) -> str:
        """Generate graph view containers HTML."""
        views = []
        for i, model in enumerate(graphs_data.keys()):
            active = 'active' if i == 0 else ''
            views.append(
                f'<div id="graph-{model}" class="graph-view {active}" '
                f'style="width: 100%; height: 700px;"></div>'
            )
        return '\n'.join(views)
    
    def _generate_stats_cards(self, stats_data: Dict) -> str:
        """Generate initial stats card HTML."""
        # Will be populated by JavaScript
        return '<h3>Select a model</h3>'
    
    def _create_single_graph_figure(self, G: nx.DiGraph, model_name: str):
        """Create a single graph figure for a model."""
        fig = go.Figure()
        
        if G.number_of_nodes() == 0:
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        # Use spring layout for better visualization
        try:
            pos = nx.spring_layout(G, k=1.0/np.sqrt(G.number_of_nodes()), iterations=50, seed=42)
        except:
            pos = nx.spring_layout(G, iterations=50, seed=42)
        
        # Create edge traces grouped by anomaly score
        edge_x_high = []
        edge_y_high = []
        edge_x_med = []
        edge_y_med = []
        edge_x_low = []
        edge_y_low = []
        
        for u, v, attrs in G.edges(data=True):
            if u not in pos or v not in pos:
                continue
                
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            score = attrs.get('anomaly_score', 0.0)
            
            # Categorize by score
            if score > 0.8:
                edge_x_high.extend([x0, x1, None])
                edge_y_high.extend([y0, y1, None])
            elif score > 0.5:
                edge_x_med.extend([x0, x1, None])
                edge_y_med.extend([y0, y1, None])
            else:
                edge_x_low.extend([x0, x1, None])
                edge_y_low.extend([y0, y1, None])
        
        # Add edge traces
        if edge_x_high:
            fig.add_trace(
                go.Scatter(
                    x=edge_x_high, y=edge_y_high,
                    mode='lines',
                    line=dict(width=2, color='#FF6B6B'),
                    hoverinfo='skip',
                    showlegend=False,
                    name='High Anomaly'
                )
            )
        
        if edge_x_med:
            fig.add_trace(
                go.Scatter(
                    x=edge_x_med, y=edge_y_med,
                    mode='lines',
                    line=dict(width=1.5, color='#FFA500'),
                    hoverinfo='skip',
                    showlegend=False,
                    name='Medium Anomaly'
                )
            )
        
        if edge_x_low:
            fig.add_trace(
                go.Scatter(
                    x=edge_x_low, y=edge_y_low,
                    mode='lines',
                    line=dict(width=0.5, color='#D3D3D3'),
                    hoverinfo='skip',
                    showlegend=False,
                    name='Low Anomaly'
                )
            )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            if node not in pos:
                continue
                
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            attrs = G.nodes[node]
            node_type = attrs.get('node_type', attrs.get('type', 'unknown'))
            importance = attrs.get('importance', 0.0)
            name = attrs.get('name', str(node))
            
            # Truncate long names
            if len(name) > 40:
                name = name[:37] + '...'
            
            node_text.append(
                f"<b>{name}</b><br>"
                f"Type: {node_type}<br>"
                f"Importance: {importance:.4f}<br>"
                f"In-Degree: {G.in_degree(node)}<br>"
                f"Out-Degree: {G.out_degree(node)}"
            )
            
            # Color by type
            type_colors = {
                'process': '#FF6B6B',  # Red
                'file': '#4ECDC4',     # Teal
                'network': '#95E1D3',  # Light green
                'socket': '#F38181',   # Pink
                'unknown': '#AAAAAA'   # Gray
            }
            node_colors.append(type_colors.get(node_type, '#AAAAAA'))
            
            # Size by importance
            size = 15 + importance * 40
            node_sizes.append(min(size, 50))
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            showlegend=False
        )
        
        fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{model_name.upper()} Attack Graph</b><br>"
                     f"<sub>{G.number_of_nodes()} nodes, {G.number_of_edges()} edges</sub>",
                x=0.5,
                xanchor='center',
                font=dict(size=24)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#f7fafc',
            height=700
        )
        
        return fig
    
    def export_summary_json(
        self,
        model_graphs: Dict[str, nx.DiGraph],
        model_paths: Dict[str, List[Dict]],
        model_stats: Dict[str, Dict]
    ):
        """Export summary in JSON format."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name in model_graphs.keys():
            G = model_graphs[model_name]
            paths = model_paths.get(model_name, [])
            stats = model_stats.get(model_name, {})
            
            summary['models'][model_name] = {
                'graph_stats': {
                    'num_nodes': G.number_of_nodes(),
                    'num_edges': G.number_of_edges(),
                    'num_connected_components': nx.number_weakly_connected_components(G),
                    'density': nx.density(G)
                },
                'top_paths': paths[:5],  # Top 5 paths
                'anomaly_stats': stats
            }
        
        json_path = self.output_dir / "attack_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary JSON: {json_path}")
    
    def export_graphml(self, model_graphs: Dict[str, nx.DiGraph]):
        """Export attack graphs in GraphML format (for Gephi, Cytoscape)."""
        for model_name, G in model_graphs.items():
            graphml_path = self.output_dir / f"{model_name}_attack_graph.graphml"
            nx.write_graphml(G, str(graphml_path))
            logger.info(f"Saved GraphML: {graphml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and compare attack graphs from multiple models"
    )
    
    parser.add_argument(
        '--artifacts-dir',
        type=str,
        default='artifacts',
        help='Artifacts directory with model outputs'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['magic', 'kairos', 'orthrus', 'threatrace', 'continuum_fl'],
        help='Models to visualize'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/attack_graph_visualization',
        help='Output directory for visualizations'
    )
    
    parser.add_argument(
        '--threshold-percentile',
        type=float,
        default=95.0,
        help='Percentile threshold for anomalies (default: 95.0)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Number of top anomalies to include (default: 100)'
    )
    
    parser.add_argument(
        '--top-paths',
        type=int,
        default=10,
        help='Number of top attack paths to extract (default: 10)'
    )
    
    parser.add_argument(
        '--cluster-by',
        type=str,
        choices=['entity', 'temporal', 'path'],
        default='entity',
        help='Clustering strategy (default: entity)'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Skip auto-opening browser (useful for remote servers)'
    )
    
    parser.add_argument(
        '--serve',
        action='store_true',
        help='Start HTTP server for remote access (recommended for VS Code Remote)'
    )
    
    args = parser.parse_args()
    
    # Setup
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track if HTTP server is started
    server_port_to_keep = None
    
    logger.info("=" * 80)
    logger.info("Attack Graph Visualization and Comparison")
    logger.info("=" * 80)
    
    # Process each model
    model_graphs = {}
    model_paths = {}
    model_scores = {}
    model_stats = {}
    
    for model_name in args.models:
        logger.info(f"\nProcessing model: {model_name}")
        logger.info("-" * 80)
        
        # Load model inference output
        inference_path = artifacts_dir / model_name / 'model_inference' / 'output.pkl'
        
        if not inference_path.exists():
            logger.warning(f"No inference output found for {model_name}, skipping")
            continue
        
        logger.info(f"Loading inference from: {inference_path}")
        with open(inference_path, 'rb') as f:
            inference_output = pickle.load(f)
        
        scores = np.array(inference_output['scores'])
        model_scores[model_name] = scores
        
        # Statistics
        model_stats[model_name] = {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'percentile_95': float(np.percentile(scores, 95)),
            'percentile_99': float(np.percentile(scores, 99)),
            'num_scores': len(scores)
        }
        
        logger.info(f"Scores: {len(scores)}, Range: [{scores.min():.6f}, {scores.max():.6f}]")
        
        # Load graph data from graph_transformation output
        graph_path = artifacts_dir / model_name / 'graph_transformation' / 'output.pkl'
        
        if not graph_path.exists():
            logger.warning(f"No graph data found for {model_name}, skipping")
            continue
        
        logger.info(f"Loading graph from: {graph_path}")
        with open(graph_path, 'rb') as f:
            graph_output = pickle.load(f)
        
        # Extract graph_data
        if 'graph_data' in graph_output:
            graph_data = graph_output['graph_data']
        else:
            logger.warning(f"No graph_data in output for {model_name}, skipping")
            continue
        
        # Reconstruct attack graph
        try:
            reconstructor = AttackGraphReconstructor(
                graph_data=graph_data,
                anomaly_scores=scores,
                threshold_percentile=args.threshold_percentile,
                max_path_length=10
            )
            
            attack_graph = reconstructor.reconstruct_attack_graph(
                top_k=args.top_k,
                cluster_by=args.cluster_by
            )
            
            model_graphs[model_name] = attack_graph
            
            # Extract attack paths
            paths = reconstructor.extract_attack_paths(
                attack_graph=attack_graph,
                top_k=args.top_paths
            )
            
            model_paths[model_name] = paths
            logger.info(f"Extracted {len(paths)} attack paths")
            
        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}", exc_info=True)
            continue
    
    if not model_graphs:
        logger.error("No models processed successfully")
        return 1
    
    # Create visualizations
    logger.info("\n" + "=" * 80)
    logger.info("Creating Visualizations")
    logger.info("=" * 80)
    
    visualizer = MultiModelVisualizer(output_dir)
    
    # Interactive comparison
    if PLOTLY_AVAILABLE:
        html_path = visualizer.create_interactive_comparison(
            model_graphs=model_graphs,
            model_paths=model_paths,
            model_scores=model_scores
        )
        if html_path:
            logger.info(f"✓ Interactive visualization: {html_path}")
            
            browser_opened = False
            
            # Check if we should force HTTP server mode
            if args.serve:
                logger.info("HTTP server mode requested (--serve flag)")
                browser_opened = False
            # Auto-open in browser (unless disabled)
            elif not args.no_browser:
                logger.info("Opening visualization in browser...")
                browser_opened = open_in_browser(Path(html_path))
                
                if browser_opened:
                    logger.info("✓ Opened visualization in default browser")
                else:
                    logger.warning("Could not auto-open browser (likely running on remote server)")
            
            # If browser didn't open or serve mode requested, start HTTP server
            if not browser_opened or args.serve:
                logger.info("\n" + "-" * 80)
                logger.info("Starting HTTP server for remote access...")
                logger.info("-" * 80)
                
                server_port = start_http_server(output_dir, port=8000)
                
                if server_port:
                    logger.info("\n" + "=" * 80)
                    logger.info("🌐 HTTP SERVER STARTED")
                    logger.info("=" * 80)
                    logger.info(f"\nVisualization ready at: http://localhost:{server_port}")
                    logger.info(f"\n📋 To access the visualization:")
                    logger.info(f"\n  1. In VS Code, check the PORTS panel (bottom, next to TERMINAL)")
                    logger.info(f"     → VS Code should auto-detect port {server_port}")
                    logger.info(f"\n  2. Click the globe icon 🌐 next to port {server_port}")
                    logger.info(f"     → This opens the visualization directly in your browser!")
                    logger.info(f"\n  3. If port not auto-detected:")
                    logger.info(f"     → Click 'Forward a Port' and enter: {server_port}")
                    logger.info(f"     → Then click the globe icon 🌐")
                    logger.info(f"\n  💡 The page will open directly to the visualization")
                    logger.info(f"  ⏳ Server keeps running until you press Ctrl+C")
                    logger.info("\n" + "=" * 80 + "\n")
                    
                    # Store server port to keep server running
                    server_port_to_keep = server_port
                else:
                    logger.error("Failed to start HTTP server")
                    logger.info(f"\nManual access options:")
                    logger.info(f"  1. Download the HTML file to your local machine")
                    logger.info(f"  2. Use VS Code 'Open Preview' on the HTML file")
                    logger.info(f"  3. Path: {html_path}")

    
    # Export summary JSON
    visualizer.export_summary_json(
        model_graphs=model_graphs,
        model_paths=model_paths,
        model_stats=model_stats
    )
    
    # Export GraphML
    visualizer.export_graphml(model_graphs)
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info(f"  • Interactive viewer: {output_dir}/attack_graph_viewer.html")
    logger.info(f"  • Attack summary: {output_dir}/attack_summary.json")
    logger.info(f"  • GraphML exports: {output_dir}/*.graphml")
    
    # If HTTP server is running, keep the script alive
    if server_port_to_keep:
        logger.info("\n" + "=" * 80)
        logger.info("⏳ SERVER RUNNING - Waiting for you to view the visualization...")
        logger.info("=" * 80)
        logger.info(f"\n🌐 Access at: http://localhost:{server_port_to_keep}")
        logger.info(f"   (Opens directly to the visualization)")
        logger.info("\n💡 Press Ctrl+C when done to stop the server and exit\n")
        
        try:
            # Keep the main thread alive so the daemon server thread continues
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n\n" + "=" * 80)
            logger.info("🛑 Server stopped by user")
            logger.info("=" * 80)
            logger.info("Visualization session ended.\n")
            return 0
    else:
        logger.info("\nNext steps:")
        logger.info(f"  1. Explore interactive visualization in your browser")
        logger.info(f"  2. Use navigation tabs to switch between models")
        logger.info(f"  3. Import GraphML files into Gephi/Cytoscape for advanced analysis")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
