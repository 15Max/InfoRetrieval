from graphblas import Matrix, Vector, dtypes, agg, binary, monoid, semiring
import numpy as np
import time



n = 2_000_000
density = 1e-5           # only 0.0001% of possible edges
nnz    = int(n*n*density)

rows  = np.random.randint(0, n, size=nnz)
cols  = np.random.randint(0, n, size=nnz)
vals  = np.random.rand(nnz)

print(len(vals))



start_time = time.perf_counter()

RandomMatrix = Matrix.from_coo(rows.tolist(), cols.tolist(), vals.tolist(), dup_op = binary.plus, dtype=dtypes.FP32)

row_sums = RandomMatrix.reduce_rowwise("plus")
row_sums = row_sums.new()
indices, values = row_sums.to_coo()
inv = Vector(row_sums.dtype, row_sums.size)
for i, s in zip(indices.tolist(), values.tolist()):
    if s != 0.0:
        inv[i] = 1.0 / s
D = inv.diag()
RandomMatrix = D.mxm(RandomMatrix, op = 'plus_times')

print("Standardized")

vector = np.full(shape = RandomMatrix.nrows, fill_value = 1/RandomMatrix.nrows)
R = Vector.from_dense(vector,  dtype = dtypes.FP32)

teleport_vector = Vector.from_dense(np.full(shape = RandomMatrix.nrows, fill_value = 1/RandomMatrix.nrows), dtype = dtypes.FP32)


diff = 10e6
iterations = 0
max_iters = 200

damping_factor = 0.85

while diff > 10e-6 and iterations < max_iters:
    old_R = R
    R = damping_factor * old_R.vxm(RandomMatrix) + (1 - damping_factor) * teleport_vector
    diff = (R - old_R).reduce(agg.L1norm)
    iterations += 1

end_time = time.perf_counter() - start_time


# Extract all non-zero entries
indices, values = R.to_coo()

# Get the indices that would sort values descending
order = np.argsort(values)[::-1][:10]

print("Top 10 PageRank scores:")
for idx in order:
    print(f"  node {indices[idx]}: {values[idx]:.6f}")

print(f"Iterations: {iterations} in time: {end_time}")