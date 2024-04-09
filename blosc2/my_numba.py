from time import time
import numba as nb
import numpy as np
import blosc2

# Define a function named mean_numba decorated with @nb.jit to compile the code with Numba
# nopython=True and parallel=True specify that nopython mode will be used and
# the loop will attempt to be parallelized with Numba

def mean_numba(a, b, c):
    out = np.empty(a.shape, a.dtype)  # Create an empty array to store the result
    for i in nb.prange(a.shape[0]):  # Iterate over the first dimension of array a in parallel
        for j in nb.prange(a.shape[1]):  # Iterate over the second dimension of array a in parallel
                out[i, j] = (a[i, j] + b[i, j] * c[i, j]) + 2  # Calculate the corresponding value in the resulting array
    return out  # Return the resulting array

# Create NumPy arrays a, b, and c with size 3000x4000 and values generated using linspace
dtype = np.float64  # Data type for the arrays
size = 12000000  # Total size of the arrays (3000 * 4000)
a = np.linspace(0,10,num=size,dtype=dtype).reshape(3000, 4000)  # Array a
b = np.linspace(0,10,num=size,dtype=dtype).reshape(3000, 4000)  # Array b
c = np.linspace(0,10,num=size,dtype=dtype).reshape(3000, 4000)  # Array c

# Measure the execution time of the approach with Numba
start = time()  # Initial time
mean_expr = mean_numba(a, b, c)  # Call the mean_numba function
t = time() - start  # Calculate elapsed time
print("Time Numba = {}s".format((t)))  # Print elapsed time

# Convert NumPy arrays to compressed arrays using blosc2
a1 = blosc2.asarray(a)  # Compressed array a
b1 = blosc2.asarray(b)  # Compressed array b
c1 = blosc2.asarray(c)  # Compressed array c
# Perform the mathematical operation using LazyExpr: addition, multiplication, and addition of 2
expr = a1 + b1 * c1  # LazyExpr expression
expr += 2  # Add 2
t1 = time()  # Initial time
res = expr.evaluate()  # Evaluate the LazyExpr expression and get the result
tt = time() - t1  # Calculate elapsed time
print("Time LazyExpr = {}s".format((tt)))  # Print elapsed time

# Compare the results obtained with Numba and LazyExpr using np.testing.assert_allclose
# This checks if both results are close within a tolerance
np.testing.assert_allclose(res[:], mean_expr)
