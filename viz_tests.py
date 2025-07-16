from Viz import plot_subgraph
import PageRank


def main():
    # Test the plot_subgraph function
    method = "matrix"

    wiki_pr = PageRank.WikiPageRank(damping_factor= 0.85, max_iterations=100, tolerance=1e-6)
    wiki_pr.load_data("data/wiki-topcats.txt", 
                      "data/wiki-topcats-page-names.txt", 
                      "data/wiki-topcats-categories.txt" 
                      )

    #plot_subgraph(wiki_pr, top_k=15, with_labels=True, label_count=5, save_path="results/test.png", title="Test Subgraph")

# Compute Ancient Greek category PR and visualize
    pagerank_scores_Greek_Gods = PageRank.run_and_report(
        wiki_pr, 
        "results/PR_Greek_Gods_results.csv",
        category="Category:Greek_gods",
        method=method
    )

    wiki_pr.pagerank_scores = pagerank_scores_Greek_Gods
    plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=8, save_path="results/Greek_Gods.png", title="Top 20 nodes in Greek Gods by PageRank Score")

# Compute Ancient Greek legendary creatures category PR and visualize
    pagerank_scores_Greek_Legendary_Creatures = PageRank.run_and_report(
        wiki_pr,
        "results/PR_Greek_Legendary_Creatures_results.csv",
        category="Category:Greek_legendary_creatures",
        method=method
    )

    wiki_pr.pagerank_scores = pagerank_scores_Greek_Legendary_Creatures
    plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=8, save_path="results/Greek_Legendary_Creatures.png", title="Top 20 nodes in Greek Legendary Creatures by PageRank Score")


# Personalized Pagerank for the greek gods and legendary creatures
    weights = [0.6, 0.4]
    personalized_scores = PageRank.personalized_pagerank(
        pagerank_dicts=[pagerank_scores_Greek_Gods, pagerank_scores_Greek_Legendary_Creatures],
        weights=weights
    )
    

    wiki_pr.pagerank_scores = personalized_scores
    wiki_pr.save_results("results/wiki_pagerank_personalized_results.csv")
    plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=10, save_path="results/personalized_Greek.png", title="Top Personalized Greek Nodes by PageRank Score", node_sparsity=2.0)

if __name__ == "__main__":
    main()