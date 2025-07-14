# This file contains the implementation of the PageRank algorithm on the dataset https://snap.stanford.edu/data/wiki-topcats.html
# We create a class WikiPageRank that encapsulates the PageRank algorithm and provides methods to load the dataset, compute PageRank scores, 
# analyze results, and visualize the distribution of scores. Then we supply other utility functions to load, combine, and analyze PageRank scores.


import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from graphblas import Matrix, Vector, binary, agg, dtypes
import matplotlib.pyplot as plt
import time
import os

class WikiPageRank:
    """
    PageRank implementation for the wiki-topcats network dataset.
    """
    
    def __init__(self, damping_factor: float = 0.85, max_iterations: int = 100, 
                 tolerance: float = 1e-6):
        """
        Initialize PageRank parameters.
        
        Args:
            damping_factor: Probability of following a link (typically 0.85)
            max_iterations: Maximum number of iterations
            tolerance: Convergence threshold
        """
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Graph representation
        self.graph = defaultdict(list)  # adjacency list, node_id -> list of neighbors
        self.nodes = set()
        self.page_names = {}  # node_id -> page_name
        self.categories = {}  # category -> list of nodes
        self.node_categories = defaultdict(list)  # node -> list of categories

        # Dangling & outdegree tracking
        self.out_degrees = {}  # node_id -> outdegree
        self.dangling_nodes = []  # list of dangling nodes

        # Matrix
        self.M = None


        # Results
        self.pagerank_scores = {}
        self.iterations_taken = 0
        self.converged = False
    
    def load_data(self, graph_file: str, page_names_file: str, 
                  categories_file: str = None):
        """
        Load the wiki-topcats dataset.
        
        Args:
            graph_file: Path to wiki-topcats file (edges)
            page_names_file: Path to wiki-topcats-page-names file
            categories_file: Path to wiki-topcats-categories file (optional)
        """
        print("Loading graph edges...")
        self._load_graph(graph_file)
        
        print("Loading page names...")
        self._load_page_names(page_names_file)
        
        if categories_file:
            print("Loading categories...")
            self._load_categories(categories_file)

        print("Caching out‐degrees and dangling nodes...")
        self._compute_degrees()
        
        print(f"Loaded graph with {len(self.nodes)} nodes and {self._count_edges()} edges")
    
    def _load_graph(self, graph_file: str):
        """
        Load the graph edges from the wiki-topcats file.
        
        Args:
            graph_file: Path to the wiki-topcats file containing edges
        """
        with open(graph_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        source = int(parts[0])
                        target = int(parts[1])
                        self.graph[source].append(target)
                        self.nodes.add(source)
                        self.nodes.add(target)
    
    def _load_page_names(self, page_names_file: str):
        """
        Load page names from the wiki-topcats-page-names file.
        
        Args:
            page_names_file: Path to the wiki-topcats-page-names file.
        """
        with open(page_names_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)  # Split on first space only
                    if len(parts) >= 2:
                        node_id = int(parts[0])
                        page_name = parts[1]
                        self.page_names[node_id] = page_name
    
    def _load_categories(self, categories_file: str):
        """
        Load categories from the wiki-topcats-categories file.
        
        Args:
            categories_file: Path to the wiki-topcats-categories file.
        """
        with open(categories_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(';')
                    if len(parts) >= 2:
                        category = parts[0].strip()
                        node_ids = [int(x) for x in parts[1].split()]
                        self.categories[category] = node_ids
                        
                        # Build reverse mapping
                        for node_id in node_ids:
                            self.node_categories[node_id].append(category)
    
    def _build_global_matrix(self):
        """
        Build and cache the n×n row-stochastic adjacency matrix M
        assuming node IDs are 0,1,…,n-1.
        """
        
        # Sanity check, remove me after
        n = len(self.nodes)
        assert self.nodes == set(range(n)), "Expected nodes == {0,…,n-1}"

        # Populate the matrix, 0 when no outgoing edge, 1/|outdegree| otherwise
        rows, cols, vals = [], [], []
        for u in range(n):
            outdeg = self.out_degrees.get(u, 0)
            if outdeg == 0:
                continue
            w = 1.0 / outdeg
            for v in self.graph[u]:
                rows.append(u)
                cols.append(v)
                vals.append(w)

        # Create the matrix object
        self.M = Matrix.from_coo(
            rows, cols, vals,
            dup_op=binary.plus, # Really no need i think
            dtype=dtypes.FP32
        )


    
    def _count_edges(self) -> int:
        """Count total number of edges in the graph."""
        return sum(len(neighbors) for neighbors in self.graph.values())
    
    def _compute_degrees(self):
        """Compute and cache out‐degrees and dangling node list."""
        self.out_degrees = {u: len(self.graph[u]) for u in self.nodes}
        self.dangling_nodes = [u for u, d in self.out_degrees.items() if d == 0]
    
    def compute_pagerank(self, method: str = 'matrix', category : Optional[str] = None) -> Dict[int, float]:
        """
        Compute PageRank scores using specified method.
        
        Args:
            method: 'power_iteration' or 'matrix' (for smaller graphs)
            category: Optional category to perform Topic-Specific PageRank computation
        
        Returns:
            Dictionary mapping node_id to PageRank score
        """
        if method == 'power_iteration':
            return self._pagerank_power_iteration(category = category)
        elif method == 'matrix':
            return self._pagerank_matrix(category = category)
        else:
            raise ValueError("Method must be 'power_iteration' or 'matrix'")
    
    def _pagerank_power_iteration(self, category: Optional[str] = None) -> Dict[int, float]:
        """
        Compute PageRank using power iteration method.
        More memory efficient for large graphs.

        Args:
            category: Optional category to perform Topic-Specific PageRank computation
        
        Returns:
            Dictionary mapping node_id to PageRank score
        """
        print("Computing PageRank using power iteration...")
        start_time = time.perf_counter()

        # If we pass a category, we need to define the positives entries of the teleport vector only on such nodes and 0 otherwise.
        # If no category is passed, we use a uniform distribution over all nodes.

        if category is not None:
            if category not in self.categories:
                raise ValueError(f"Unknown category {category!r}")
            topic_nodes = set(self.categories[category])
            n_nodes = len(topic_nodes)
            teleport_vector = {node : (1 / n_nodes if node in topic_nodes else 0.0) for node in self.nodes}
        else:
            n_nodes = len(self.nodes)
            teleport_vector = {node : (1 / n_nodes) for node in self.nodes}  # Uniform distribution
        
        
        
        current_scores = {node : teleport_vector[node] for node in self.nodes} # R0
        next_scores = {node: 0.0 for node in self.nodes}

        
        for iteration in range(self.max_iterations):
            # Reset next scores
            for node in self.nodes:
                # Teleportation contribution
                next_scores[node] = (1 - self.damping_factor) * teleport_vector[node] 
            
            # Catch the leaked probabilities from dangling nodes
            leaked_probs = sum(current_scores[node] for node in self.dangling_nodes) 
            
            # And add it back
            for node in self.nodes:
                next_scores[node] += self.damping_factor * leaked_probs * teleport_vector[node]
            
            # Add contributions from regular nodes
            for source in self.nodes:
                if self.out_degrees[source] > 0:
                    contribution = self.damping_factor * current_scores[source] / self.out_degrees[source]
                    for target in self.graph[source]:
                        if target in self.nodes:
                            next_scores[target] += contribution
            
            # Check for convergence
            diff = sum(abs(next_scores[node] - current_scores[node]) 
                      for node in self.nodes)
            
            if diff < self.tolerance:
                self.converged = True
                self.iterations_taken = iteration + 1
                break
            
            # Swap scores for next iteration
            current_scores, next_scores = next_scores, current_scores
        
        else:
            self.converged = False
            self.iterations_taken = self.max_iterations
        
        self.pagerank_scores = current_scores
        
        elapsed_time = time.perf_counter() - start_time
        print(f"PageRank computation completed in {elapsed_time:.2f} seconds")
        print(f"Converged: {self.converged}, Iterations: {self.iterations_taken}")
        
        return self.pagerank_scores
    
    def _pagerank_matrix(self, category: Optional[str] = None) -> Dict[int, float]:
        """
        Compute PageRank using matrix method and GraphBLAS
        """
        print("Computing PageRank using matrix method leveraging GraphBLAS...")

        diff, iterations, max_iterations = 10e6, 0, 200

        if self.M is None:
            self._build_global_matrix()
            print(f"Done creating the matrix of shape: {self.M.shape}")

        start_time = time.perf_counter()

        if category is not None:
            if category not in self.categories:
                raise ValueError(f"Unknown category {category!r}")
            topic_nodes = set(self.categories[category])
            n_nodes = len(topic_nodes)
            node_values = {node : (1 / n_nodes if node in topic_nodes else 0.0) for node in self.nodes}
            teleport_vector = Vector.from_dict(node_values, dtype = dtypes.FP32)
        else:
            n_nodes = len(self.nodes)
            node_values = {node : (1 / n_nodes) for node in self.nodes}  # Uniform distribution
            teleport_vector = Vector.from_dict(node_values, dtype = dtypes.FP32)

        R = Vector.from_dict(node_values, dtype = dtypes.FP32)

        while diff > self.tolerance and iterations < max_iterations:
            old_R = R
            R = self.damping_factor * old_R.vxm(self.M) + (1 - self.damping_factor) * teleport_vector
            diff = (R - old_R).reduce(agg.L1norm)
            iterations += 1
        
        if diff < self.tolerance:
                self.converged = True

        self.iterations_taken = iterations

        elapsed = time.perf_counter() - start_time
        print(f"PageRank completed in {elapsed:.2f}s")
        print(f"Converged: {self.converged}, Iterations: {self.iterations_taken}")

        
        self.pagerank_scores = R.to_dict()

        return self.pagerank_scores
        
        
    def get_top_pages(self, n: int = 10) -> List[Tuple[int, str, float]]:
        """
        Get top N pages by PageRank score.
        
        Args:
            n: Number of top pages to return
            
        Returns:
            List of (node_id, page_name, pagerank_score) tuples
        """
        if not self.pagerank_scores:
            raise ValueError("PageRank scores not computed yet. Call compute_pagerank() first.")
        
        # Sort by PageRank score in descending order
        sorted_scores = sorted(self.pagerank_scores.items(), 
                             key=lambda x: x[1], reverse=True)
        
        top_pages = []
        for i in range(min(n, len(sorted_scores))):
            node_id, score = sorted_scores[i]
            page_name = self.page_names.get(node_id, f"Unknown_{node_id}")
            top_pages.append((node_id, page_name, score))
        
        return top_pages
    
    def analyze_results(self, top_n: int = 20):
        """
        Analyze and display PageRank results.
        
        Args:
            top_n: Number of top pages to display
        """
        if not self.pagerank_scores:
            raise ValueError("PageRank scores not computed yet. Call compute_pagerank() first.")
        
        print(f"\n=== PageRank Analysis ===")
        print(f"Graph Statistics:")
        print(f"  Nodes: {len(self.nodes):,}")
        print(f"  Edges: {self._count_edges():,}")
        print(f"  Convergence: {self.converged}")
        print(f"  Iterations: {self.iterations_taken}")
        
        # Basic statistics
        scores = list(self.pagerank_scores.values())
        print(f"\nPageRank Score Statistics:")
        print(f"  Mean: {np.mean(scores):.6f}")
        print(f"  Std:  {np.std(scores):.6f}")
        print(f"  Min:  {np.min(scores):.6f}")
        print(f"  Max:  {np.max(scores):.6f}")
        
        # Top pages
        top_pages = self.get_top_pages(top_n)
        print(f"\nTop {top_n} Pages by PageRank Score:")
        print("-" * 80)
        for i, (node_id, page_name, score) in enumerate(top_pages, 1):
            print(f"{i:2d}. {page_name:<50} {score:.6f}")
        
        # Category analysis if available
        if self.categories:
            self._analyze_categories(top_pages)
    
    def _analyze_categories(self, top_pages: List[Tuple[int, str, float]]):
        """Analyze top pages by categories."""
        print(f"\nCategory Analysis for Top Pages:")
        print("-" * 50)
        
        for i, (node_id, page_name, score) in enumerate(top_pages[:10], 1):
            categories = self.node_categories.get(node_id, [])
            if categories:
                print(f"{i:2d}. {page_name}")
                for cat in categories[:3]:  # Show first 3 categories
                    print(f"    - {cat}")
                if len(categories) > 3:
                    print(f"    ... and {len(categories) - 3} more")
            else:
                print(f"{i:2d}. {page_name} (no categories)")
            print()
    
    def plot_pagerank_distribution(self, bins: int = 50):
        """
        Plot the distribution of PageRank scores.
        
        Args:
            bins: Number of bins for histogram
        """
        if not self.pagerank_scores:
            raise ValueError("PageRank scores not computed yet. Call compute_pagerank() first.")
        
        scores = list(self.pagerank_scores.values())
        
        plt.figure(figsize=(12, 5))
        
        # Linear scale histogram
        plt.subplot(1, 2, 1)
        plt.hist(scores, bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel('PageRank Score')
        plt.ylabel('Frequency')
        plt.title('PageRank Score Distribution')
        plt.grid(True, alpha=0.3)
        
        # Log scale histogram
        plt.subplot(1, 2, 2)
        plt.hist(scores, bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel('PageRank Score')
        plt.ylabel('Frequency')
        plt.title('PageRank Score Distribution (Log Scale)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_file: str):
        """
        Save PageRank results to a file.
        
        Args:
            output_file: Path to output file
        """
        # Did we compute PageRank scores?
        if not self.pagerank_scores:
            raise ValueError("PageRank scores not computed yet. Call compute_pagerank() first.")

        # Does the file already exist?
        if os.path.exists(output_file):
            # Warn the user 
            print(f"Warning: {output_file} already exists. Results will be overwritten.")

        # Write results to CSV
        with open(output_file, 'w') as f:
            f.write("node_id,page_name,pagerank_score\n")
            
            # Sort by PageRank score
            sorted_scores = sorted(self.pagerank_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
            
            for node_id, score in sorted_scores:
                page_name = self.page_names.get(node_id, f"Unknown_{node_id}")
                # Escape commas in page names
                page_name = page_name.replace(',', ';')
                f.write(f"{node_id},{page_name},{score:.8f}\n")
        
        print(f"Results saved to {output_file}")

    
    def analyze_pagerank_scores(self, top_n: int = 20):
        """
        Analyze the PageRank scores of the WikiPageRank instance.
        Args:
            top_n: Number of top pages to display
        """
        
        self.analyze_results(top_n = top_n)
        self.plot_pagerank_distribution()
        top_10 = self.get_top_pages(n = 10)

        print("\nTop 10 pages:")
        for rank, (node_id, page_name, score) in enumerate(top_10, 1):
            print(f"{rank}. {page_name}: {score:.6f}")


def load_pagerank_scores(file_path: str) -> dict:
    """
    Load a CSV file containing PageRank scores and return a dictionary
    mapping node_id to pagerank_score.

    The CSV is expected to have columns:
      - node_id
      - page_name (ignored)
      - pagerank_score

    Args:
        file_path: Path to the CSV file.

    Returns:
        A dict where keys are node_id (int) and values are pagerank_score (float).
    """
    # Read only the columns we need
    df = pd.read_csv(file_path, usecols=['node_id', 'pagerank_score'])
    
    # Convert to dict mapping node_id to pagerank_score
    return dict(zip(df['node_id'].astype(int), df['pagerank_score'].astype(float)))



def personalized_pagerank(pagerank_dicts : List[Dict[int, float]], weights: List[float]) -> Dict[int, float]:
    """
    Combine several pagerank score dicts (node -> score) into one,
    using the given convex weights.

    Args:
        pagerank_dicts: List of dictionaries containing PageRank scores.
        weights: List of weights for each PageRank dict, must sum to 1 (we're doing a linear combination).
    
    Returns:
        A dictionary mapping node_id to combined PageRank score.
    """
    if len(pagerank_dicts) != len(weights):
        raise ValueError("Need as many weight as pagerank dicts")
    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Weights must sum to 1")

    combined = defaultdict(float)
    for w, pr in zip(weights, pagerank_dicts):
        for node, score in pr.items():
            combined[node] += w * score

    # Normalize the combined scores to ensure they sum to 1 (kills small numerical errors)
    total = sum(combined.values())
    if total > 0:
        for node in combined:
            combined[node] /= total

    return dict(combined)


def execute_or_load_pagerank(WikiPageRank: WikiPageRank, output_file: str, category: Optional[str] = None, method: str = 'matrix') -> Dict[int, float]:
    """
    Execute PageRank computation or load existing results.
    
    Args:
        WikiPageRank: An instance of the WikiPageRank class.
        output_file: Path to the output file for PageRank results.
        category: Optional category to filter PageRank computation, only used if we're computing PageRank.
        method: Method to compute PageRank, either 'power_iteration' or 'matrix'.
    
    Returns:
        Dictionary of PageRank scores.
    """
    if os.path.exists(output_file):
        print(f"Loading existing PageRank results from {output_file}...")
        return load_pagerank_scores(output_file)
    else:
        print("Computing PageRank scores...")
        pagerank_scores = WikiPageRank.compute_pagerank(method = method, category = category)
        WikiPageRank.save_results(output_file)
        return pagerank_scores


def run_and_report(pr_obj : WikiPageRank, file_name : str, category: Optional[str] = None, method: str = 'matrix') -> Dict[int, float]:
    """
    Run PageRank computation and report results. If file_name exists, load results instead and analyze them.

    Args:
        pr_obj: An instance of the WikiPageRank class.
        file_name: Path to the output file for PageRank results or to a file to load results from.
        category: Optional category to filter PageRank computation, only used if we're computing PageRank.
        method: Method to compute PageRank, either 'power_iteration' or 'matrix'.
    
    Returns:
        Dictionary of PageRank scores.
    """
    scores = execute_or_load_pagerank(WikiPageRank = pr_obj, output_file = file_name, category = category, method = method)
    pr_obj.pagerank_scores = scores
    if category != None:
        print(f"--- {category} ---")
    else:
        print(f"--- Wiki PageRank ---")
    pr_obj.analyze_pagerank_scores()
    return scores

# Example usage
def main():
    """
    Example usage of the WikiPageRank class.
    """


    wiki_pr = WikiPageRank(damping_factor=0.85, max_iterations=100, tolerance=1e-6)
    wiki_pr.load_data(
        graph_file="data/wiki-topcats.txt",
        page_names_file="data/wiki-topcats-page-names.txt",
        categories_file="data/wiki-topcats-categories.txt"
    )


    _ = run_and_report(wiki_pr, "results/wiki_pagerank_results.csv")

    pagerank_scores_RNA = run_and_report(wiki_pr, "results/wiki_pagerank_RNA_results.csv", category="Category:RNA")
    
    pagerank_scores_BIO = run_and_report(wiki_pr, "results/wiki_pagerank_BIO_results.csv", category="Category:Biomolecules")

    # Could be interesting to study what happens when categories are very distinct (cover different nodes)
    # Could be interesting to see if there exist some "bridge categories" that happen to connect them when we combine the PRs

    weights = [0.3, 0.7] 

    personalized_scores = personalized_pagerank(
        pagerank_dicts=[pagerank_scores_RNA, pagerank_scores_BIO],
        weights=weights
    )

    wiki_pr.pagerank_scores = personalized_scores
    print("\n\n--- Personalized PageRank (RNA + BIO) ---\n\n")
    wiki_pr.analyze_pagerank_scores()
    wiki_pr.save_results("results/wiki_pagerank_personalized(RNA+BIO)_results.csv")
    
if __name__ == "__main__":
    main()