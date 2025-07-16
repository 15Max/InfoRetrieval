# Information Retrieval final project: PageRank
This repository contains the implementation of the PageRank algorithm on the wikipedia top categories dataset, freely available at: https://snap.stanford.edu/data/wiki-topcats.html .

### Requirements
All requirements are listed in the `requirements.txt` file. Or if you prefer, you can create a conda environment using the provided `conda.yml` file.

To run the "gugol" interface, first run PageRank.py and the notebook embedder.py; otherwise, the necessary files will not be generated.
Then, run the command fastapi run gugol_main.py. The default port should be 8000.
For more instructions on changing the port if necessary, refer to the FastAPI documentation available online.
*Note*: The "gugol" interface is intended for the project presentation only. We do not expect others to run it, so we will not provide further documentation on how to use this interface.


