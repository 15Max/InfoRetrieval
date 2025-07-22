
#-----------------------------------------------------------------------------------------------------
# IMPORTANT: In order to run this code, you need to have the following files in the "data" directory:
# - embeddings.npy: Precomputed embeddings for the dataset.
# - pagerank_scores.npy: Precomputed PageRank scores for the dataset.
# - wiki-topcats-categories.txt: File containing categories for each node.
# - wiki_pagerank_RNA_results.csv: File containing page names and their corresponding node
# You can change the last file to match your pageRank scores and categories.
# Run PageRank.py and the notebook embedder.ipynb to generate these files.
#-----------------------------------------------------------------------------------------------------

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

#Load model and data
device = "cpu"
print(f"Using device: {device}")
print("Loading model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.to(device)
print("Model loaded successfully.")
embeddings_np = np.load("data/embeddings.npy")
embeddings = torch.tensor(embeddings_np)
embeddings = embeddings.to(device)
print("Embeddings loaded successfully.")
pagerank_scores = torch.tensor(np.load("data/pagerank_scores.npy"))
pagerank_scores = pagerank_scores.to(device)
print("Pagerank scores loaded successfully.")
df = pd.read_csv("results/wiki_pagerank_results.csv")
df.sort_values(by="node_id", inplace=True)
page_names = df["page_name"].tolist()
print("Page names loaded successfully.")

#Load categories from file
categories = {}
node_categories = {}
with open("data/wiki-topcats-categories.txt", "r") as f:
    for line in f:
                line = line.strip()
                if line:
                    parts = line.split(';')
                    if len(parts) >= 2:
                        category = parts[0].strip()
                        #Remove the "Category:" prefix if it exists
                        if category.startswith("Category:"):
                            category = category[len("Category:"):].strip()
                            
                        node_ids = [int(x) for x in parts[1].split()]
                        categories[category] = node_ids
                        
                        # Build reverse mapping
                        for node_id in node_ids:
                            if int(node_id) not in node_categories:
                                node_categories[int(node_id)] = []
                            node_categories[int(node_id)].append(category)
print("Categories loaded successfully.")
print("STARTING SERVER...")


class SimilarityReturn(BaseModel):
    title: str
    categories: list[str]
    rank: int
    score: float

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/similarity", response_model=list[SimilarityReturn])
async def calculate_similarity(text: str, top_k: int = 10, similarity_threshold: float = 0.0, pagerank_weight: float = 0.99) -> list[SimilarityReturn]:
    """Calculate similarity scores for a given text against precomputed embeddings.
    Args:
        text (str): The input text to compare against embeddings.
        top_k (int): Number of top results to return.
        similarity_threshold (float): Minimum similarity score to consider a result valid.
        pagerank_weight (float): Weight of the pagerank score in the final result. Default is 0.99.        
    Returns:
        dict: A list of dictionaries containing the title, categories, rank, and score of the top similar items.
    """
    print(f"Calculating emebdding for text: {text}")
    text_embedding = model.encode(text, device=device)
    print("Text embedding calculated.")
    text_embedding = torch.tensor(text_embedding).to(device)
    print("Calculating similarities...")
    similarities = model.similarity(text_embedding, embeddings)
    print("Similarities calculated.")
    final_scores = pagerank_weight * pagerank_scores + (1 - pagerank_weight) * similarities[0]
    threshold_mask = similarities[0] > similarity_threshold
    final_scores_with_threshold = final_scores[threshold_mask]
    sorted_final_indices_with_threshold = final_scores_with_threshold.argsort(descending=True)
    results = []
    rank = 1
    for i in sorted_final_indices_with_threshold[:top_k]:
        idx = threshold_mask.nonzero(as_tuple=True)[0][i].item()
        title = page_names[idx]
        score = final_scores_with_threshold[i].item()
        categories = node_categories.get(idx, [])
        results.append(SimilarityReturn(title=title, categories=categories, rank=rank, score=score))
        rank += 1
    print(f"Returning {len(results)} results.")           
         
    return results

    
