# Information Retrieval final project: PageRank
This repository contains the implementation of the PageRank algorithm on the wikipedia top categories dataset, freely available at: https://snap.stanford.edu/data/wiki-topcats.html .

### Requirements
All requirements are listed in the `requirements.txt` file. Or if you prefer, you can create a conda environment using the provided `conda.yml` file.


### PageRank
We implemented the Pagerank algorithm, leveraging two different data structures: an adjacency list and an adjacency matrix. 
You can choose which of these two methods to use by setting the `method` variable in the `PageRank.py` file to either `list` or `matrix`.
The main difference is that for the `matrix` method we leveraged the Python  [GraphBLAS](https://graphblas.org/) library (a Python wrapper for the C API).  This allows for parallel computation, leading to a significant speedup in the computation of the PageRank scores.

### Visualizations
We also added some visualizations of the web graph, colored by the corresponding PageRank score.

You can choose which visualization to display by modifying the `viz_tests.py` file, and choosing a category of interest by browsing the ones [available.](data/wiki-topcats-categories.txt)

To avoid too much clutter we plotted only a small subset of the nodes, and also analyzed results for topic specific and personalized PageRank on different categories.

### Gugol Interface
We also implemented a simple interface to query .... #todo: continue

To run the "gugol" interface, first run `PageRank.py` and the notebook `embedder.py`; otherwise, the necessary files will not be generated.
Then, run the command `fastapi run gugol_main.py`. The default port should be 8000.
For more instructions on changing the port if necessary, refer to the FastAPI documentation available online.
*Note*: The "gugol" interface is intended for the project presentation only. We do not expect others to run it, so we will not provide further documentation on how to use this interface.


