import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import PageRank
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_subgraph(
    pr_obj,
    top_k: int = 30,
    with_labels: bool = False,
    label_count: int = 5,
    save_path: str = None, 
    title: str = None, 
    node_sparsity: float = 1.5
):
    """
    Plot a subgraph of the PageRank results.
    Works with global PageRank or a single category

    Args:
        pr_obj: WikiPageRank instance with pagerank_scores computed.
        top_k: Number of top nodes by PageRank score to include in the subgraph.
        with_labels: Whether to annotate nodes with their page names.
        label_count: Number of top nodes (by PageRank score) to annotate.
        save_path: If provided, saves the figure to this file path.
        title: Custom title for the plot. If None, a default title is used.
    """
    if not pr_obj.pagerank_scores:
        raise ValueError("PageRank scores not computed. Run compute_pagerank() first.")

    # Determine which nodes to plot based on the presence of categories
 
    candidate_nodes = pr_obj.nodes
    if title is None:
        title = f"Top {top_k} Nodes by Pagerank Score"

    
    # Pick top_k by PageRank score
    top_nodes = sorted(
        candidate_nodes, key=lambda n: pr_obj.pagerank_scores.get(n, 0), reverse=True
    )[:top_k]

    # Build subgraph from top nodes
    G = nx.DiGraph()
    for node in top_nodes:
        G.add_node(node)
        for target in pr_obj.graph.get(node, []):
            if target in top_nodes:
                G.add_edge(node, target)

    # Prepare visualization data 
    pr_scores = [pr_obj.pagerank_scores.get(n, 0.0) for n in G.nodes()]
    pos = nx.spring_layout(G, seed=123, k=node_sparsity) # k controls the distance between nodes

    plt.figure(figsize=(12, 8))
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=pr_scores,
        cmap=cm.viridis,
        edgecolors='black',
        linewidths=0.8,
        node_size=500
    )
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        edge_color='black',
        width=0.5,
        alpha=0.35, # Transparency of edges
        arrowsize=10,
        connectionstyle='arc3,rad=0.1'
    )

    # Labels can be added if requested
    if with_labels and label_count > 0:
        ax = plt.gca()
        top_nodes_for_labels = sorted(
            G.nodes(), key=lambda n: pr_obj.pagerank_scores.get(n, 0), reverse=True
        )[:label_count]
        for node in top_nodes_for_labels:
            x, y = pos[node]
            label = pr_obj.page_names.get(node, str(node))
            ax.text(x, y + 0.08, label, fontsize=9, fontweight='bold', ha='center', va='bottom')

    plt.title(title, fontsize=14, fontweight='bold')
    cbar = plt.colorbar(nodes)
    cbar.set_label("PageRank Score")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show(block=False)
        plt.pause(0.001)  # Allow the plot to render without blocking
        plt.close('all')


def safe_filename(name: str) -> str:
    # Replace anything that is not alphanumeric, dash, or underscore
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name)


def plot_combinations(
    wiki_pr,
    categories,
    method="matrix",
    weights=None,
    top_k=20,
    with_labels=True,
    label_count=20,
    node_sparsity=1.5
):
    if weights is None:
        weights = [0.5, 0.5]

    if len(weights) != len(categories):
        raise ValueError(
            f"Length of weights ({len(weights)}) must match number of categories ({len(categories)})"
        )

    pagerank_dicts = []

    for category in categories:
        safe_cat = safe_filename(category)
        csv_path = f"results/{safe_cat}.csv"
        img_path = f"results/{safe_cat}.png"
        title = f"Top {top_k} nodes in {category} by PageRank Score"

        pr_scores = PageRank.run_and_report(
            wiki_pr,
            csv_path,
            category=category,
            method=method
        )
        pagerank_dicts.append(pr_scores)

        wiki_pr.pagerank_scores = pr_scores
        plot_subgraph(
            wiki_pr,
            top_k=top_k,
            with_labels=with_labels,
            label_count=label_count,
            save_path=img_path,
            title=title,
            node_sparsity=node_sparsity
        )

    combined_scores = PageRank.personalized_pagerank(
        pagerank_dicts=pagerank_dicts,
        weights=weights
    )

    wiki_pr.pagerank_scores = combined_scores
    combined_name = "personalized_" + "_".join(safe_filename(cat) for cat in categories)
    combined_csv = f"results/{combined_name}.csv"
    combined_img = f"results/{combined_name}.png"

    wiki_pr.save_results(combined_csv)
    combined_title = f"Top {top_k} nodes combined ({', '.join(categories)}) by Personalized PageRank Score"

    plot_subgraph(
        wiki_pr,
        top_k=top_k,
        with_labels=with_labels,
        label_count=label_count,
        save_path=combined_img,
        title=combined_title,
        node_sparsity=node_sparsity
    )


def main():

    # Initialize WikiPageRank object
    wiki_pr = PageRank.WikiPageRank(
        damping_factor=0.85,
        max_iterations=100,
        tolerance=1e-6
    )
    
    # Load the dataset
    wiki_pr.load_data(
        "data/wiki-topcats.txt",
        "data/wiki-topcats-page-names.txt",
        "data/wiki-topcats-categories.txt"
    )
    # Compute general PageRank and visualize
    PageRank.run_and_report(wiki_pr, "results/wiki_pagerank_results.csv", method="matrix")
    plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=20, save_path="results/general_small.png")
    plot_subgraph(wiki_pr, top_k=100, with_labels=False, label_count=20, save_path="results/general_large.png", node_sparsity=1.5)

    # RNA and Biomolecules categories
    plot_combinations(
    wiki_pr=wiki_pr,
    categories=["Category:RNA", "Category:Biomolecules"],
    method="matrix",
    weights=[0.3, 0.7],
    top_k=20,
    with_labels=True,
    label_count=20,
    node_sparsity=1.5
    )

    # Ancient Greek Gods and Ancient Greek Legendary creatures
    plot_combinations(
        wiki_pr=wiki_pr,
        categories=["Category:Greek_gods", "Category:Greek_legendary_creatures"],
        method="matrix",
        weights=[0.5, 0.5],
        top_k=20,
        with_labels=True,
        label_count=20,
        node_sparsity=1.5
    )

    # Artificial Intelligence and Multiplayer Video Games
    plot_combinations(
        wiki_pr=wiki_pr,
        categories=["Category:Artificial_intelligence", "Category:Multiplayer_video_games"],
        method="matrix",
        weights=[0.6, 0.4],
        top_k=20,
        with_labels=True,
        label_count=20,
        node_sparsity=1.5
    )




if __name__ == "__main__":
    main()
