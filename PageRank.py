# This file contains the implementation of the PageRank algorithm. The PageRank algorithm is used to rank web pages based on their importance,
# which is determined by a function that takes into account the links between pages. Essentially, a page rank is obtained by a propagation of
# the ranks of the pages that are linked to it. The following in a brief theory reminder (a "teacher" comment)
#
# Let M be an adjacency matrix of size n x n, where n is the number of web pages. F_u is the set of forward links from page u, B_u the set of backward link
# to page u, c a normalization factor and E(u) a positive vector of size n that is used to allow escaping dead ends. The PageRank of a page u is given by the formula:
# 
# R(u) = c * [sum_{v in B_u} (R(v) / |F_v|) + E(u)]
#
# To compute this, we will use the power iteration method, which is an iterative algorithm that computes the dominant eigenvector of a matrix. This is
# possible because the PageRank matrix is a stochastic matrix and therefore by the Perron-Frobenius theorem, it has a unique dominant eigenvector which
# corresponds to the PageRank of the pages. Another way to see the vector R is the stationary distribution of a Markov chain where the states
# are the web pages and the transition probabilities are given by the links between the pages. 
#
# Power Iteration algorithm
# If we have A, a diagonalizable matrix, say n x n, then we can decompose it using eigendecomposition as A = VΛV^{-1} with V is a matrix where the columns are
# the eigenvectors of A and Λ is a diagonal matrix containing the sorted (in abs) eigenvalues of A. Now observe that this holds:
#
# A = VΛV^{-1}
# AA = VΛV^{-1}VΛV^{-1} = VΛ^{2}V^{-1}
# AAA = VΛV^{-1}VΛV^{-1}VΛV^{-1} = VΛ^{3}V^{-1}
# ...
# A^k = {VΛV^{-1}}^k = VΛ^{k}V^{-1}
#
# Now, given b a random vector of size n, by definition, it can always be rewritten as a linear combination of the eigenvectors of A: b = Vg
# where g is another vector of size n. Consider now:
#
# A^k b = A^k Vg = VΛ^{k}V^{-1} Vg = VΛ^{k}g = \sum_{i = 1}^n v_i λ_i^k g_i 
#
# Taking out the first eigenvalue (the biggest in abs) we get:
#
# λ_1 \sum_{i = 1}^n v_i (λ_i / λ_1)^k g_i
#
# Now, as k → ∞, note that (λ_i / λ_1)^k goes to 0 so we proved that A^k b converges to v_1 λ_1^k g_1. Normalize it using the ∥ ∥2 and we've just obtained
# the unique dominant eigenvector for A^k aka the stationary distribution: the probability of being in a page after a "long time" has passed. Of course, we don't
# set k to ∞, we stop when we reach a certain tolerance.
#
# Do note that for us A = c*M + (1-c)Ex1 (E uniform vector,1 a vector full of ones and c h)



import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import time

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
        
        # Data structures
        # Maybe we can join dictionaries (adj list, page_names and node_categories) into one?
        # Also, do we need to store the node category? (Topic PR)

        self.graph = defaultdict(list)  # adjacency list
        self.nodes = set()
        self.page_names = {}  # node_id -> page_name
        self.categories = {}  # category -> list of nodes
        self.node_categories = defaultdict(list)  # node -> list of categories
        
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
        
        print(f"Loaded graph with {len(self.nodes)} nodes and {self._count_edges()} edges")
    
    def _load_graph(self, graph_file: str):
        """Load the graph edges from the wiki-topcats file."""
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
        """Load page names from the wiki-topcats-page-names file."""
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
        """Load categories from the wiki-topcats-categories file."""
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
    
    def _count_edges(self) -> int:
        """Count total number of edges in the graph."""
        return sum(len(neighbors) for neighbors in self.graph.values())
    
    def compute_pagerank(self, method: str = 'power_iteration') -> Dict[int, float]:
        """
        Compute PageRank scores using specified method.
        
        Args:
            method: 'power_iteration' or 'matrix' (for smaller graphs)
            
        Returns:
            Dictionary mapping node_id to PageRank score
        """
        if method == 'power_iteration':
            return self._pagerank_power_iteration()
        elif method == 'matrix':
            return self._pagerank_matrix()
        else:
            raise ValueError("Method must be 'power_iteration' or 'matrix'")
    
    def _pagerank_power_iteration(self) -> Dict[int, float]:
        """
        Compute PageRank using power iteration method.
        More memory efficient for large graphs.
        """
        print("Computing PageRank using power iteration...")
        start_time = time.perf_counter()
        
        # Initialize PageRank scores
        n_nodes = len(self.nodes)
        initial_score = 1.0 / n_nodes
        
        # Current and next iteration scores
        current_scores = {node: initial_score for node in self.nodes}
        next_scores = {node: 0.0 for node in self.nodes}
        
        # Precompute out-degrees for efficiency
        out_degrees = {node: len(self.graph[node]) for node in self.nodes}
        
        # Handle dangling nodes (nodes with no outgoing links)
        dangling_nodes = [node for node in self.nodes if out_degrees[node] == 0]
        
        for iteration in range(self.max_iterations):
            # Reset next scores
            for node in self.nodes:
                next_scores[node] = (1 - self.damping_factor) / n_nodes
            
            # Add contributions from dangling nodes
            dangling_sum = sum(current_scores[node] for node in dangling_nodes)
            dangling_contribution = self.damping_factor * dangling_sum / n_nodes
            
            for node in self.nodes:
                next_scores[node] += dangling_contribution
            
            # Add contributions from regular nodes
            for source in self.nodes:
                if out_degrees[source] > 0:
                    contribution = self.damping_factor * current_scores[source] / out_degrees[source]
                    for target in self.graph[source]:
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
    
    def _pagerank_matrix(self) -> Dict[int, float]:
        """
        Compute PageRank using matrix method.
        Only recommended for smaller graphs due to memory requirements.
        """
        print("Computing PageRank using matrix method...")
        start_time = time.time()
        
        # Create node index mapping
        node_list = list(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        n_nodes = len(node_list)
        
        # Build transition matrix
        M = np.zeros((n_nodes, n_nodes))
        
        for source in self.nodes:
            source_idx = node_to_idx[source]
            out_degree = len(self.graph[source])
            
            if out_degree > 0:
                for target in self.graph[source]:
                    target_idx = node_to_idx[target]
                    M[target_idx, source_idx] = 1.0 / out_degree
            else:
                # Dangling node - distribute equally to all nodes
                M[:, source_idx] = 1.0 / n_nodes
        
        # Apply damping factor
        M = self.damping_factor * M + (1 - self.damping_factor) / n_nodes
        
        # Power iteration
        v = np.ones(n_nodes) / n_nodes
        
        for iteration in range(self.max_iterations):
            v_new = M @ v
            
            # Check convergence
            if np.linalg.norm(v_new - v, 1) < self.tolerance:
                self.converged = True
                self.iterations_taken = iteration + 1
                break
            
            v = v_new
        else:
            self.converged = False
            self.iterations_taken = self.max_iterations
        
        # Convert back to dictionary
        self.pagerank_scores = {node_list[i]: v[i] for i in range(n_nodes)}
        
        elapsed_time = time.time() - start_time
        print(f"PageRank computation completed in {elapsed_time:.2f} seconds")
        print(f"Converged: {self.converged}, Iterations: {self.iterations_taken}")
        
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
        if not self.pagerank_scores:
            raise ValueError("PageRank scores not computed yet. Call compute_pagerank() first.")
        
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

# Example usage
def main():
    """
    Example usage of the WikiPageRank class.
    """
    # Initialize PageRank calculator
    wiki_pr = WikiPageRank(damping_factor=0.85, max_iterations=100, tolerance=1e-6)
    
    # Load data (adjust file paths as needed)
    wiki_pr.load_data(
        graph_file="data/wiki-topcats.txt",
        page_names_file="data/wiki-topcats-page-names.txt",
        categories_file="data/wiki-topcats-categories.txt"  # Optional
    )
    
    # Compute PageRank
    pagerank_scores = wiki_pr.compute_pagerank(method='power_iteration')
    
    # Analyze results
    wiki_pr.analyze_results(top_n=20)
    
    # Plot distribution
    wiki_pr.plot_pagerank_distribution()
    
    # Save results
    wiki_pr.save_results("wiki_pagerank_results.csv")
    
    # Get specific results
    top_10 = wiki_pr.get_top_pages(10)
    print("\nTop 10 pages:")
    for rank, (node_id, page_name, score) in enumerate(top_10, 1):
        print(f"{rank}. {page_name}: {score:.6f}")

if __name__ == "__main__":
    main()