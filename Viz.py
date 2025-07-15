import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

def plot_subgraph(
    pr_obj,
    top_k: int = 30,
    with_labels: bool = False,
    label_count: int = 5,
    save_path: str = None, 
    title: str = None
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
    pos = nx.spring_layout(G, seed=123, k=1.5) # k controls the distance between nodes

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
        alpha=0.7,
        arrowsize=12,
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
        plt.show()
