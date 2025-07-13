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

def pagerank(webpages: np.ndarray, teleport_vector: np.ndarray, tolerance: float = 1e-12, max_iterations: int = 100_000) -> tuple:
    """
    PageRank algorithm without damping
    :param webpages: np.ndarray, adjacency matrix of the web pages
    :param teleport_vector: np.ndarray, vector used to allow escaping dead ends
    :param tolerance: np.int32, the tolerance for convergence
    :return: tuple of the PageRank vector and the number of iterations
    """

    # Vars to track the convergence and iterations
    diff = 10e6
    iterations = 0 

    # Initialize the PageRank vector with a uniform distribution
    R = np.full(shape = webpages.shape[0], fill_value = 1/webpages.shape[0], dtype = float)

    # Power iteration method following the original paper pseudocode so without the damping factor. This implies that
    # we can handle dangling nodes by redistributing the leaked probability mass back to R^{i+1} but  if no dangling nodes are present we never teleport!
    # Thisit allows cyclic behaviors, aka the surfer gets stucked in a loop of pages forever (Bottom strongly connected component) -> unrealistic

    while diff > tolerance and iterations < max_iterations:
        old_R = R
        R = old_R @ webpages
        d = np.linalg.norm(old_R, 1) - np.linalg.norm(R, 1) # This is the leaked probability mass (aka )
        R =  R + d*teleport_vector # Redistribute the leaked probability mass back to the matrix
        diff = np.linalg.norm(R - old_R) 
        iterations += 1

    return R, iterations


def pagerank_damping(webpages: np.ndarray, teleport_vector: np.ndarray, tolerance: float = 1e-12, damping_factor: float = 0.85) -> tuple:
    """
    PageRank algorithm with damping
    :param webpages: np.ndarray, adjacency matrix of the web pages 
    :param teleport_vector: np.ndarray, vector used to allow escaping dead ends
    :param tolerance: np.int32, the tolerance for convergence
    :param damping_factor: np.float16, the damping factor (default is 0.85)
    :return: tuple of the PageRank vector and the number of iterations
    """

    # Vars to track the convergence and iterations 
    diff = 10e6
    iterations = 0

    # Initialize the PageRank vector
    R = np.full(shape = webpages.shape[0], fill_value = 1/webpages.shape[0], dtype = float) 

    # Power iteration method, this time we use the damping factor: we can always teleport with probability 1 - damping_factor and this 
    # handles both dangling nodes and cyclic behaviors, aka the surfer can always escape the loop of pages.

    while diff > tolerance:
        old_R = R
        R = damping_factor * (old_R @ webpages) + (1 - damping_factor) * teleport_vector
        diff = np.linalg.norm(R - old_R)
        iterations += 1

    return R, iterations




if __name__ == "__main__":

    random_matrix = np.random.rand(5,5)
    random_matrix /= np.sum(random_matrix, axis = 1, keepdims = True)
    random_uniform_vector = np.full(shape = random_matrix.shape[0], fill_value = 1 / random_matrix.shape[0])

    R, iterations = pagerank(webpages = random_matrix,
                             teleport_vector = random_uniform_vector)
    

    print(random_matrix)
    print(f"Using PageRank without damping, we achieved R stationary distribution:{R} in this number of iterations: {iterations}, it sums to: {np.sum(R)}")

    R, iterations = pagerank_damping(webpages = random_matrix,
                                     teleport_vector = random_uniform_vector)
    
    print(f"Using PageRank with damping, we achieved R stationary distribution:{R} in this number of iterations: {iterations}, it sums to: {np.sum(R)}")


    trap_matrix = np.zeros((5,5))
    # 1→2
    trap_matrix[0, 1] = 1.0
    # 2→3
    trap_matrix[1, 2] = 1.0
    # 3→4
    trap_matrix[2, 3] = 1.0
    # 4→5
    trap_matrix[4, 3] = 1.0
    # 5→4
    trap_matrix[3, 4] = 1.0

    print(trap_matrix)

    R, iterations = pagerank(webpages = trap_matrix,
                             teleport_vector = random_uniform_vector)
    
    print(f"Using PageRank without damping, we achieved R stationary distribution:{R} in this number of iterations: {iterations}, it sums to: {np.sum(R)}")

    R, iterations = pagerank_damping(webpages = trap_matrix,
                                     teleport_vector = random_uniform_vector)
    
    print(f"Using PageRank with damping, we achieved R stationary distribution:{R} in this number of iterations: {iterations}, it sums to: {np.sum(R)}")

