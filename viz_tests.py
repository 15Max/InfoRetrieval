from Viz import plot_subgraph
import PageRank


def main():
    # Test the plot_subgraph function
    method = "matrix"

    # Choose theme categories to visualize, set to False to skip
    ancient_greece = True
    artificial_intelligence = True
    biology = True
    fantasy = True

    wiki_pr = PageRank.WikiPageRank(damping_factor= 0.85, max_iterations=100, tolerance=1e-6)
    wiki_pr.load_data("data/wiki-topcats.txt", 
                      "data/wiki-topcats-page-names.txt", 
                      "data/wiki-topcats-categories.txt" 
                      )

    
    # Compute general PageRank and visualize
    PageRank.run_and_report(wiki_pr, "results/wiki_pagerank_results.csv", method=method)

    plot_subgraph(wiki_pr, top_k=100, with_labels=False, label_count=5, save_path="results/general_large.png", title="Pagerank ", node_sparsity=1.5)
    plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=10, save_path="results/general_small.png", title="Top 20 nodes by PageRank Score")

    if biology:
        # Compute RNA category PR and visualize
        pagerank_scores_RNA = PageRank.run_and_report(
            wiki_pr,
            "results/wiki_pagerank_RNA_results.csv",
            category="Category:RNA",
            method=method
        )
        
        wiki_pr.pagerank_scores = pagerank_scores_RNA
        plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=8, save_path="results/RNA_20.png", title="Top 20 RNA Nodes by PageRank Score")

        # Compute BIO category PR and visualize
        pagerank_scores_BIO = PageRank.run_and_report(
            wiki_pr,
            "results/wiki_pagerank_BIO_results.csv",
            category="Category:Biomolecules",
            method=method
        )
    
        wiki_pr.pagerank_scores = pagerank_scores_BIO
        plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=8, save_path="results/BIO_20.png", title="Top 20BIO Nodes by PageRank Score")

        # Personalized PageRank combining RNA and BIO 
        weights = [0.3, 0.7]
        personalized_scores = PageRank.personalized_pagerank(
            pagerank_dicts=[pagerank_scores_RNA, pagerank_scores_BIO],
            weights=weights
        )
    
        wiki_pr.pagerank_scores = personalized_scores
        wiki_pr.save_results("results/wiki_pagerank_personalized(RNA+BIO)_results.csv")
 
        plot_subgraph(wiki_pr, top_k=30, with_labels=True, label_count=10, save_path="results/personalized.png", title="Top 30 RNA + BIO Nodes by Personalized PageRank Score")
    if ancient_greece:
       
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
        plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=10, save_path="results/personalized_Greek.png", title="Top 20 Personalized Greek Nodes by PageRank Score", node_sparsity=2.0)

    if fantasy:
        # Compute Fantasy category PR and visualize
        pagerank_scores_Fantasy = PageRank.run_and_report(
            wiki_pr, 
            "results/PR_Fantasy_results.csv",
            category="Category:Fantasy_books_by_series", 
            method=method
        )

        wiki_pr.pagerank_scores = pagerank_scores_Fantasy
        plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=8, save_path="results/Fantasy.png", title="Top 20 nodes in Fantasy books by PageRank Score")

        pagerank_scores_Fantasy_Authors = PageRank.run_and_report(
            wiki_pr, 
            "results/PR_Fantasy_Authors_results.csv",
            category="Category:English_fantasy_writers",
            method=method
        )
        wiki_pr.pagerank_scores = pagerank_scores_Fantasy_Authors
        plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=8, save_path="results/Fantasy_Authors.png", title="Top 20 nodes in English Fantasy Authors by PageRank Score")

        # Personalized Pagerank for the fantasy books and authors
        weights = [0.5, 0.5]
        personalized_scores = PageRank.personalized_pagerank(
            pagerank_dicts=[pagerank_scores_Fantasy, pagerank_scores_Fantasy_Authors],
            weights=weights,
        )
        wiki_pr.pagerank_scores = personalized_scores
        wiki_pr.save_results("results/wiki_pagerank_personalized_fantasy_results.csv")

        plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=10, save_path="results/personalized_Fantasy.png", title="Top 20 Personalized Fantasy books and Authors Nodes by PageRank Score", node_sparsity=2.0)

    if artificial_intelligence:
        # Compute Artificial Intelligence category PR and visualize
        pagerank_scores_AI = PageRank.run_and_report(
            wiki_pr, 
            "results/PR_Artificial_Intelligence_results.csv",
            category="Category:Artificial_intelligence",
            method=method
        )

        wiki_pr.pagerank_scores = pagerank_scores_AI
        plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=8, save_path="results/AI.png", title="Top 20 nodes in Artificial Intelligence by PageRank Score", node_sparsity=2.0)

        pagerank_scores_AI_researchers = PageRank.run_and_report(
            wiki_pr, 
            "results/PR_Researchers_results.csv",
            category="Category:Artificial_intelligence_researchers",
            method=method
        )
        wiki_pr.pagerank_scores = pagerank_scores_AI_researchers
        plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=8, save_path="results/Researchers.png", title="Top 20 nodes in Researchers by PageRank Score", node_sparsity=2.0)

        # Compute Robotics category PR and visualize
        pagerank_scores_Robotics = PageRank.run_and_report(
            wiki_pr, 
            "results/PR_Robotics_results.csv",
            category="Category:Robotics",
            method=method
        )   
        wiki_pr.pagerank_scores = pagerank_scores_Robotics
        plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=8, save_path="results/Robotics.png", title="Top 20 nodes in Robotics by PageRank Score", node_sparsity=2.0)

        # Personalized Pagerank for the AI and Researchers
        weights = [0.5, 0.5]
        personalized_scores = PageRank.personalized_pagerank(
            pagerank_dicts=[pagerank_scores_AI, pagerank_scores_AI_researchers],
            weights=weights,
        )           

        wiki_pr.pagerank_scores = personalized_scores
        wiki_pr.save_results("results/wiki_pagerank_personalized_AI_results.csv")
        plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=10, save_path="results/personalized_AI.png", title="Top 20 Personalized AI + Researchers Nodes by PageRank Score", node_sparsity=2.0)

        if fantasy:
            # Putting two completely different categories together
            weights = [0.5, 0.5]
            personalized_scores = PageRank.personalized_pagerank(
                pagerank_dicts=[pagerank_scores_Fantasy, pagerank_scores_AI],
                weights=weights,
            )
            wiki_pr.pagerank_scores = personalized_scores
            wiki_pr.save_results("results/wiki_pagerank_personalized_Fantasy_AI_results.csv")
            plot_subgraph(wiki_pr, top_k=20, with_labels=True, label_count=10, save_path="results/personalized_Fantasy_AI.png", title="Top 20 Personalized Fantasy + AI Nodes by PageRank Score", node_sparsity=2.0)



if __name__ == "__main__":
    main()