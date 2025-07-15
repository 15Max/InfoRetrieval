# This file contains the code to run a small example where we compare what happens when running on a graph with a 
# bottom strongly connected component (BSCC) the "classic" PageRank algorithm with no damping and no jump vector versus the PageRank algorithm that used
# damping and a jump vector. For reference, a BSCC is a subset of a graph from which we can never escape once we enter it and without damping (and a jump vector)
# all the probability mass will eventually end up in that BSCC (limiting distribution for DTMCs)


from graphblas import Matrix, Vector, dtypes, agg
import numpy as np


def random_stochastic_square_matrix(nrows : int = 5):
    """
    Generate a random stochastic matrix of dimension nrows x nrows

    Args:
        nrows (int): The number of rows and columns in the matrix
    """

    random_matrix = np.random.rand(nrows, nrows)
    random_matrix /= np.sum(random_matrix, axis=1, keepdims=True)

    return random_matrix



def pagerank(webpages : np.ndarray, tolerance: float = 10e-6, max_iterations: int = 200):
    """
    PageRank algorithm without damping
    
    Args:
        webpages (np.ndarray): The adjacency matrix of the graph
        tolerance (float): The convergence tolerance
        max_iterations (int): The maximum number of iterations to run
    Returns:
        Vector: The PageRank vector containing the scores
        int: The number of iterations it took
    """

    # Convergence/Iterations variables
    iterations, diff, converged = 0, 10e6, False

    # Build the adjacency matrix and the initial R vector as an uniform
    A = Matrix.from_dense(webpages)
    R_values = np.full(shape = A.nrows, fill_value = 1/A.nrows)
    R = Vector.from_dense(R_values,  dtype = dtypes.FP32)

    # Teleport vector is always uniform so we just copy R
    E = R

    # Power iteration method no damping
    while converged == False and iterations < max_iterations:

        old_R = R 
        R = old_R.vxm(A)
        d = R.reduce(agg.L1norm) - old_R.reduce(agg.L1norm) # Grab leaked probability mass
        R = R + (d * E) # Add it back
        diff = (R - old_R).reduce(agg.L1norm) # Did we converge?
        if (diff < tolerance):
            converged = True
        iterations += 1

    R = R.new() # To return Vector and not VectorExpression

    return R, iterations, converged


def pagerank_damping(webpages : np.ndarray, tolerance: float = 10e-6, max_iterations: int = 200, damping_factor: float = 0.85):
    """
    PageRank algorithm with damping

    Args:
        webpages (np.ndarray): The adjacency matrix of the graph
        tolerance (float): The convergence tolerance
        max_iterations (int): The maximum number of iterations to run
        damping_factor (float): The damping factor (default is 0.85)
    Returns:
        Vector: The PageRank vector containg the scores
        int: The number of iterations it took
    """

    # Convergence/Iterations variables
    iterations, diff, converged = 0, 10e6, False

    # Build the adjacency matrix and the initial R vector as an uniform
    A = Matrix.from_dense(webpages)
    R_values = np.full(shape = A.nrows, fill_value = 1/A.nrows)
    R = Vector.from_dense(R_values,  dtype = dtypes.FP32)

    # Teleport vector is always uniform so we just copy R
    E = R

    # Power iteration method with damping
    while converged == False and iterations < max_iterations:

        old_R = R 
        R = damping_factor * old_R.vxm(A) + (1 - damping_factor) * E
        diff = (R - old_R).reduce(agg.L1norm) # Did we converge?
        if (diff < tolerance):
            converged = True
        iterations += 1

    R = R.new() # To return Vector and not VectorExpression

    return R, iterations, converged


def report_print(method: str, R_vector: Vector, iterations: int, convergence: bool = False):
    """
    Pretty print function to report the results of the PageRank algorithm

    Args:
        method (str): The name of the method used
        R_vector (Vector): The PageRank vector containing the scores
        iterations (int): The number of iterations it took to converge
        convergence (bool): Whether the algorithm converged or not
    """
    print(f"Convergence: {convergence} in this number of iterations: {iterations} using {method}, we achieved the following PageRank scores:\n {R_vector}\n\n")



# Need a visualization function or maybe do it by hand..


def main():
    """
    Main function to run the tests
    """
    
    # 1→2, 2→3, 3→4, 5→4 and 4→5
    trap_matrix = np.zeros((5,5))
    trap_matrix[0, 1] = 1.0
    trap_matrix[1, 2] = 1.0
    trap_matrix[2, 3] = 1.0
    trap_matrix[4, 3] = 1.0
    trap_matrix[3, 4] = 1.0

    random_matrix = random_stochastic_square_matrix(5)

    R, iters, converged = pagerank(random_matrix)
    report_print("PageRank no damping on random stochastic matrix", R, iters, converged)

    # Test on trap matrix
    R, iters, converged = pagerank(trap_matrix)
    report_print("PageRank no damping on a trap matrix with a BSCC", R, iters, converged)

    # Test on random matrix with damping
    R, iters, converged = pagerank_damping(random_matrix)
    report_print("PageRank damping on a random stochastic matrix", R, iters, converged)

    R, iters, converged = pagerank_damping(trap_matrix)
    report_print("PageRank damping on a trap matrix with a BSCC", R, iters, converged)


if __name__ == "__main__":
    main()

