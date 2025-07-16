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
from Viz import plot_subgraph
from utils import URL_LIST, FILE_NAMES, download_and_extract, ensure_files_exist

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

        # Graph and nodes "metadata"
        self.page_names = {}  # node_id -> page_name
        self.categories = {}  # category -> list of nodes
        self.node_categories = defaultdict(list)  # node -> list of categories
        self.out_degrees = {}  # node_id -> outdegree

        # Nodes that have outdegree equal to 0
        self.dangling_nodes = []  # list of dangling nodes

        # Adjacency matrix
        self.M = None

        # Results
        self.pagerank_scores = {}
        self.iterations_taken = 0
        self.converged = False

    def reset_convergence_params(self):
        """
        Reset the self.iterations and self.converged params. Used when we call compute_pagerank
        to not drag previous results in other computations.
        """

        self.iterations_taken = 0
        self.converged = False
    
    def load_data(self, graph_file: str, page_names_file: str, 
                  categories_file: Optional[str] = None):
        """
        Load the wiki-topcats dataset.
        
        Args:
            graph_file: Path to wiki-topcats file (edges)
            page_names_file: Path to wiki-topcats-page-names file (node "names")
            categories_file: Path to wiki-topcats-categories file (maps categories to nodes)
        """

        # Check if the files exist
        file_found = ensure_files_exist(file_list = FILE_NAMES, directory = "data")
        if not file_found:
            print("Files not found. Downloading and extracting...")
            download_and_extract(URL_LIST, save_dir="data")

        print("Loading graph edges...")
        self._load_graph(graph_file)
        
        print("Loading page names...")
        self._load_page_names(page_names_file)
        
        # Used for topic specific pagerank and some analysis, but not strictly necessary
        if categories_file:
            print("Loading categories...")
            self._load_categories(categories_file)
        
        print(f"Loaded graph with {len(self.nodes)} nodes and {self._count_edges()} edges")
    
    def _load_graph(self, graph_file: str):
        """
        Load the graph edges from the wiki-topcats file and compute the outdegrees and the dangling nodes
        
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

        # Cache also the outdegrees and dangling nodes
        self._compute_degrees()

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
        Build and cache the (n,n) row-stochastic adjacency matrix M
        """
        
        # It is assumed that nodes go from 0 to n-1 (and this is true for the dataset that we use)
        # If that was not the case, a map would need to be built and we would iterate over that.

        n = len(self.nodes)

        # Populate the matrix, 1/n when no outgoing edge, 1/|outdegree| otherwise
        rows, cols, vals = [], [], []
        for u in range(n):
            outdeg = self.out_degrees.get(u, 0)  

            # Here we catch the dangling node (has outdegree = 0) and its row in the matrix
            # gets set to a uniform probability. This is done when we define the matrix for efficiency reasons.
            # In the other method we handle it at each iteration, but by using this approach there is no need
            # to do so. Also as a side note, in this dataset there're no dangling nodes so we should never enter here

            if outdeg == 0:
                outdeg = n # Here we assume to fix the dangling node probs with a uniform distr over nodes
                w = 1.0 / n
                for v in range(n): # "Populate" the row of the node
                    rows.append(u)
                    cols.append(v)
                    vals.append(w)
                continue

            # If instead it is not dangling, uniform distr over the reached nodes
            w = 1.0 / outdeg
            for v in self.graph[u]:
                rows.append(u)
                cols.append(v)
                vals.append(w)

        # Create the matrix graphblas object
        self.M = Matrix.from_coo(
            rows, cols, vals,
            dtype=dtypes.FP32
        )


    
    def _count_edges(self) -> int:
        """
        Count total number of edges in the graph.
        """

        return sum(len(neighbors) for neighbors in self.graph.values())
    
    def _compute_degrees(self):
        """
        Compute and cache out‐degrees and dangling node list.
        """

        self.out_degrees = {u: len(self.graph[u]) for u in self.nodes}
        self.dangling_nodes = [u for u, d in self.out_degrees.items() if d == 0]
    
    def compute_pagerank(self, method: str = 'matrix', category : Optional[str] = None) -> Dict[int, float]:
        """
        Compute PageRank scores using the power iteration method and a specified data structure
        
        Args:
            method: 'list' or 'matrix' 
            category: Optional category to perform Topic-Specific PageRank computation
        
        Returns:
            Dictionary mapping node_id to PageRank score
        """

        self.reset_convergence_params()

        if method == 'list':
            return self._pagerank_list(category = category)
        elif method == 'matrix':
            return self._pagerank_matrix(category = category)
        else:
            raise ValueError("Method must be 'list' or 'matrix'")
    
    def _pagerank_list(self, category: Optional[str] = None) -> Dict[int, float]:
        """
        Compute PageRank using the power iteration method and an adjacency list data structure (python "native" approach)

        Args:
            category: Optional category to perform Topic-Specific PageRank computation
        
        Returns:
            Dictionary mapping node_id to PageRank score
        """

        print("Computing PageRank using power iteration and adjacency lists...")


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
        
        
        # The first initialization copies the one of the teleport vector ("warm start" for topic specifc PR)

        current_scores = {node : teleport_vector[node] for node in self.nodes} # R0
        next_scores = {node: 0.0 for node in self.nodes}

        
        while self.iterations_taken < self.max_iterations and self.converged == False:
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
            diff = sum(abs(next_scores[node] - current_scores[node]) for node in self.nodes)

            # Update convergence/iter check params
            self.iterations_taken += 1
            if diff < self.tolerance:
                self.converged = True
            
            # Swap scores for next iteration
            current_scores, next_scores = next_scores, current_scores
    
        self.pagerank_scores = current_scores
        
        elapsed_time = time.perf_counter() - start_time
        print(f"PageRank computation completed in {elapsed_time:.2f} seconds")
        print(f"Converged: {self.converged}, Iterations: {self.iterations_taken}")
        
        return self.pagerank_scores
    
    def _pagerank_matrix(self, category: Optional[str] = None) -> Dict[int, float]:
        """
        Compute PageRank using matrix method and GraphBLAS.
        python-graphblas is a python wrapper over GraphBLAS a library containing highly optimized C routines for sparse linear algebra
        (e.g. fast sparse matrix-vector multiplies (our case!)), making PageRank power iterations both memory and time-efficient.
        Further info at (https://graphblas.org/) and (https://pypi.org/project/python-graphblas/)
        """
        print("Computing PageRank using matrix method leveraging GraphBLAS...")

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

        while self.iterations_taken < self.max_iterations and self.converged == False:
            old_R = R
            R = self.damping_factor * old_R.vxm(self.M) + (1 - self.damping_factor) * teleport_vector
            diff = (R - old_R).reduce(agg.L1norm)
        
            if diff < self.tolerance:
                self.converged = True

            self.iterations_taken += 1

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
        plt.show(block = False)
        plt.pause(0.001)  # Allow the plot to render without blocking
        plt.close('all')  # Close the plot to free memory
        print("PageRank score distribution plotted.")
    
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
    using the given linear combination of weights

    Args:
        pagerank_dicts: List of dictionaries containing PageRank scores.
        weights: List of weights for each PageRank dict, must sum to 1
    
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
        method: Method to compute PageRank, either 'list' or 'matrix'.
    
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
        method: Method to compute PageRank, either 'list' or 'matrix'.
    
    Returns:
        Dictionary of PageRank scores.
    """
    scores = execute_or_load_pagerank(WikiPageRank = pr_obj, output_file = file_name, category = category, method = method)
    pr_obj.pagerank_scores = scores
    if category != None:
        print(f"--- {category} ---")
    else: # We're doing "general PageRank"
        print(f"--- Wiki PageRank ---")
    pr_obj.analyze_pagerank_scores()
    return scores


def main():
    method = "matrix"
    visualize = True

    wiki_pr = WikiPageRank(damping_factor=0.85, max_iterations=100, tolerance=1e-6)
    wiki_pr.load_data(
        graph_file="data/wiki-topcats.txt",
        page_names_file="data/wiki-topcats-page-names.txt",
        categories_file="data/wiki-topcats-categories.txt"
    )

    # Compute general PageRank and visualize
    run_and_report(wiki_pr, "results/wiki_pagerank_results.csv", method=method)
    if visualize:
        plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=10, save_path="results/general.png")

    # Compute RNA category PR and visualize
    pagerank_scores_RNA = run_and_report(
        wiki_pr,
        "results/wiki_pagerank_RNA_results.csv",
        category="Category:RNA",
        method=method
    )
    if visualize:
        wiki_pr.pagerank_scores = pagerank_scores_RNA
        plot_subgraph(wiki_pr, top_k=15, with_labels=True, label_count=5, save_path="results/RNA.png", title="Top RNA Nodes by PageRank Score")

    # Compute BIO category PR and visualize
    pagerank_scores_BIO = run_and_report(
        wiki_pr,
        "results/wiki_pagerank_BIO_results.csv",
        category="Category:Biomolecules",
        method=method
    )
    if visualize:
        wiki_pr.pagerank_scores = pagerank_scores_BIO
        plot_subgraph(wiki_pr, top_k=15, with_labels=True, label_count=5, save_path="results/BIO.png", title="Top BIO Nodes by PageRank Score")

    # Personalized PageRank combining RNA and BIO 
    weights = [0.3, 0.7]
    personalized_scores = personalized_pagerank(
        pagerank_dicts=[pagerank_scores_RNA, pagerank_scores_BIO],
        weights=weights
    )
    
    wiki_pr.pagerank_scores = personalized_scores
    wiki_pr.save_results("results/wiki_pagerank_personalized(RNA+BIO)_results.csv")
    if visualize:
        plot_subgraph(wiki_pr, top_k=30, with_labels=True, label_count=10, save_path="results/personalized.png", title="Top Personalized RNA + BIO Nodes by PageRank Score")

if __name__ == "__main__":
    main()
