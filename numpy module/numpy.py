import numpy as np

# --------------------------
# Copy vs View
# --------------------------
array1 = np.array([1, 2, 3])           # Original array

array_copy = array1.copy()             # Creates a new independent copy (changes to this do NOT affect array1)
array_view = array1.view()             # Creates a view that shares memory with original (changes here WILL affect array1)

print(array1)                          # Output: [1 2 3] — the original array
print(array1.dtype)                     # Output: int64 (or platform-dependent) — type of elements
print(type(array1))                     # Output: <class 'numpy.ndarray'> — confirms it is a NumPy array
print(array_copy)                       # Output: [1 2 3] — independent copy
print(array_view)                       # Output: [1 2 3] — view sharing memory with array1

%timeit array1                           # Measures execution time of referencing array1; no computation involved

# --------------------------
# User input array
# --------------------------
l = []  # Initialize an empty list to store numbers

# Loop 4 times to collect 4 numbers
for i in range(1, 5):
    int_1 = int(input("Enter Number : "))  # Take input from the user and convert to integer
    l.append(int_1)                        # Append the number to the list

print(l)  # Print the final list
# Example input/output inside:
# Enter Number : 5
# Enter Number : 10
# Enter Number : 3
# Enter Number : 8
# Output: [5, 10, 3, 8]

# --------------------------
# Array dimensions and reshaping
# --------------------------
array_1 = np.array([1, 2, 3])  # 1D array
array_2 = np.array([[1, 2], [3, 4]])  # 2D array
array_3 = np.array([[[1, 2], [3, 4]], [[4, 5], [6, 7]]])  # 3D array
array_4 = np.array([1, 2, 3, 4], ndmin=4)  # Force 4D array
array_5 = array_3.reshape(4, 2)  # Reshape 3D array to 2D
array_6 = array_5.reshape(-1)  # Flattened view
array_7 = array_3.flatten()  # Flattened copy

print(np.resize(array_3, (4, 2)))  
# Output: [[1 2]
#          [3 4]
#          [4 5]
#          [6 7]]

print(array_1.ndim)  # Dimensions of 1D array
# Output: 1

print(array_1[array_1 > 2])  # Elements greater than 2
# Output: [3]

print(array_1[[0,1]])  # Indexing elements
# Output: [1 2]

print(array_2.ndim)  # 2D array dimensions
# Output: 2

print(array_2.shape)  # Shape of 2D array
# Output: (2, 2)

print(array_3.ndim)  # 3D array dimensions
# Output: 3

print(array_3.size)  # Total number of elements
# Output: 8

print(array_3.shape)  # Shape of 3D array
# Output: (2, 2, 2)

print(np.array_split(array_3, 3))  # Split along first axis
# Output: [array([[[1, 2],
#                 [3, 4]]]), 
#          array([[[4, 5],
#                 [6, 7]]]), 
#          array([], shape=(0, 2, 2), dtype=int64)]

print(array_4.ndim)  # 4D array dimensions
# Output: 4

print(array_5)  # Reshaped 2D array
# Output: [[1 2]
#          [3 4]
#          [4 5]
#          [6 7]]

print(array_6)  # Flattened array
# Output: [1 2 3 4 4 5 6 7]

print(array_7)  # Flattened array copy
# Output: [1 2 3 4 4 5 6 7]

# --------------------------
# Iterating over arrays
# --------------------------
for i in array_3:
    for j in i:
        for k in j:
            print(k)  # Iterating manually over each element
# Output sequence: 1 2 3 4 4 5 6 7 (each element printed separately)

for i in np.nditer(array_3):
    print(i)  # Efficient iterator over all elements
# Output sequence: 1 2 3 4 4 5 6 7

for i, d in np.ndenumerate(array_3):
    print(i, d)  # Index and value in array
# Output:
# (0,0,0) 1
# (0,0,1) 2
# (0,1,0) 3
# (0,1,1) 4
# (1,0,0) 4
# (1,0,1) 5
# (1,1,0) 6
# (1,1,1) 7

# --------------------------
# Vectorize functions
# --------------------------

# Array of names
names = np.array(['Jim', 'Luke', 'Josh', 'Pete'])

# Use vectorize to apply a function to each element
# Function: get the first character of each name
first_letter_j = np.vectorize(lambda s: s[0])(names) == 'J'
# show index number that start with 'J'
print(np.where(np.char.startswith(na, 'J')))
# Use boolean indexing to select names that start with 'J'
print(names[first_letter_j])

# --------------------------
# Array creation functions
# --------------------------
var1 = "Hello"
print(np.fromiter(var1, dtype='U2'))  
# Create a NumPy array from iterable (string here), each element can hold up to 2-character Unicode

print(np.zeros((3, 3)))  
# 3x3 array filled with zeros

print(np.ones((3, 3)))  
# 3x3 array filled with ones

print(np.empty((3, 3)))  
# 3x3 array allocated but uninitialized (values may be random)

print(np.full((3,3),7))  
# 3x3 array where every element is 7

print(np.diag([1,2,3]))  
# Create a diagonal matrix with 1,2,3 on the diagonal

print(np.arange(4))  
# Array with values from 0 up to (but not including) 4: [0 1 2 3]

print(np.eye(3))  
# 3x3 identity matrix

print(np.linspace(0, 10, num=5))  
# 5 equally spaced numbers between 0 and 10: [0. 2.5 5. 7.5 10.]

# --------------------------
# Random functions
# --------------------------
print(np.random.rand(3, 3))  
# 3x3 array of random numbers from uniform distribution [0,1)

print(np.random.randn(3, 3))  
# 3x3 array from standard normal distribution (mean=0, std=1)

print(np.random.ranf(3))  
# 1D array of 3 random numbers from uniform [0,1)

print(np.random.randint(5, 20, 5))  
# 1D array of 5 random integers between 5 (inclusive) and 20 (exclusive)

print(np.random.normal(size=3))  
# 1D array of 3 random numbers from standard normal distribution

print(np.random.uniform())  
# Single random number from uniform distribution [0,1)

print(np.random.exponential())  
# Single random number from exponential distribution (scale=1 by default)

array = np.array([1, 2, 3, 4])
np.random.shuffle(array)  # Shuffle elements in-place randomly
print(array)  # Shuffled array

# --------------------------
# Data type conversions
# --------------------------
data_type1 = np.array([1, 2, 3])  
# Default integer array

data_type2 = np.array([1, 2, 3], dtype="f")  
# Explicitly create float32 array

data_type3 = np.int_(data_type2)  
# Convert float array to integer using numpy int_ function

data_type4 = data_type3.astype(float)  
# Convert integer array back to float using astype

print(data_type1, data_type1.dtype)  
# Output array and its data type (int64 or int32 depending on system)

print(data_type2.dtype)  
# Data type of data_type2 (float32)

print(data_type3.dtype)  
# Data type after conversion to int

print(data_type4.dtype)  
# Data type after conversion back to float

# --------------------------
# Indexing and slicing
# --------------------------
var1 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])  
# 3D array with shape (1, 3, 3)

var2 = np.array([[[11, 12, 13], [14, 15, 16], [17, 18, 19]]])  
# Another 3D array with same shape

print(var1[0][1])  
# Access the 2nd row of the first (and only) block: [4, 5, 6]

print(var1[0][-2])  
# Access the 2nd last row (same as above): [4, 5, 6]

print(var2[0][0:1])  
# Slice first row (still 2D shape): [[11, 12, 13]]

print(var2[::2])  
# Take every other element along the first axis (only first block in this case)

# ------------------------------
# Advanced Slicing Tricks
# ------------------------------

# 1. All rows, only 2nd column
print("var1[0, :, 1]     :", var1[0, :, 1])    # [2, 5, 8]

# 2. Last row reversed
print("var1[0, -1, ::-1] :", var1[0, -1, ::-1])  # [9, 8, 7]

# 3. Diagonal elements (row=col)
print("Diagonal var1     :", var1[0, range(3), range(3)])  # [1, 5, 9]

# 4. Every alternate column
print("var2[0, :, ::2]   :\n", var2[0, :, ::2])
# [[11 13]
#  [14 16]
#  [17 19]]

# 5. Fancy indexing (specific elements)
print("var2 fancy        :", var2[0, [0, 2], [1, 2]])  # [12, 19]

# 6. Entire block but reversed rows
print("var1 reversed rows:\n", var1[0, ::-1])
# [[7 8 9]
#  [4 5 6]
#  [1 2 3]]

# 7. Entire block but reversed cols
print("var2 reversed cols:\n", var2[0, :, ::-1])
# [[13 12 11]
#  [16 15 14]
#  [19 18 17]]

# 8. Select sub-matrix (first 2 rows, last 2 cols)
print("Sub-matrix var1   :\n", var1[0, 0:2, 1:3])
# [[2 3]
#  [5 6]]

# --------------------------
# Concatenation & stacking
# --------------------------
print(np.concatenate((var1, var2), axis=1))  
# Concatenate arrays along axis 1 (rows inside each block)
# Result shape: (1, 6, 3)

print(np.stack((var1, var2)))  
# Stack arrays along a new axis (axis=0 by default)
# Result shape: (2, 1, 3, 3)

print(np.vstack((var1, var2)))  
# Vertical stack (row-wise concatenation)
# Result shape: (2, 3, 3)

print(np.hstack((var1, var2)))  
# Horizontal stack (column-wise concatenation)
# Result shape: (1, 3, 6)

# --------------------------
# Arithmetic operations
# --------------------------
print(var1 + 3)  # Add scalar 3 to each element

print(var1.sum())  # Sum of all elements in var1

print(np.min(var1))  # Minimum value in var1

print(np.min(var1, axis=1))  # Minimum along axis 1 (rows)

print(np.sin(var1))  # Sine of each element

print(np.cos(var1))  # Cosine of each element

print(np.exp(var1))  # Exponential of each element

print(np.argmin(var1))  # Index of minimum element in flattened array

print(np.max(var1))  # Maximum value

print(np.argmax(var1))  # Index of maximum element in flattened array

print(var1 + var2)  # Elementwise addition

print(np.add(var1, var2))  # Equivalent elementwise addition

print(var1 - var2)  # Elementwise subtraction

print(np.subtract(var1, var2))  # Equivalent subtraction

print(var1 * var2)  # Elementwise multiplication

print(np.multiply(var1, var2))  # Equivalent multiplication

print(np.dot(var1, var2))  # Dot product

print(np.vdot(var1, var2))  # Vector dot product (flattened)

print(np.inner(var1, var2))  # Inner product

print(np.outer(var1, var2))  # Outer product

print(np.cross(var1, var2))  # Cross product (only for 3-element vectors)

print(np.sqrt(var1))  # Square root of each element

print(var1 / var2)  # Elementwise division

print(np.divide(var1, var2))  # Equivalent division

print(var1 // var2)  # Floor division

print(var1 % var2)  # Modulus

print(np.mod(var1, var2))  # Equivalent modulus

print(var1 ** 2)  # Elementwise power

print(np.power(var1, 2))  # Equivalent power

print(var1.T)  # Transpose of array

print(np.absolute(var1))  # Absolute value

print(1 / var1)  # Reciprocal

print(np.reciprocal(var1))  # Equivalent reciprocal

print(np.cumsum(var1))  # Cumulative sum of elements

# --------------------------
# Queries & sorting
# --------------------------
var1 = np.array([1, 2, 2, 5, 5, 6, 9, 8, 5, 2, 5, 7, 5])
print(var1)  # Original array

query1 = np.where(var1 == 2)  # Indices where elements equal 2
query2 = np.searchsorted(var1, 11, side="right")  # Index to insert 11 to maintain order
print(query1)
print(query2)

print(np.sort(var1))  # Sorted version of array

# Unique values, their first occurrence index, and counts
print(np.unique(var1, return_index=True, return_counts=True))

# --------------------------
# Statistics
# --------------------------

arr = np.array([10, 20, 30, 40, 50])
arr2 = np.array([20,30,40,50,60])

print(np.gradient(y,x))

print("Range (Peak to Peak):", np.ptp(arr))  # Max - Min

print("Mean:", np.mean(arr))  # Average of elements

print("Median:", np.median(arr))  # Middle value

print("Standard Deviation:", np.std(arr))  # Measure of spread

print("Variance:", np.var(arr))  # Squared deviation from mean

print("Sum:", np.sum(arr))  # Sum of all elements

print("Cumulative Sum:", np.cumsum(arr))  # Running total

print("Product:", np.prod(arr))  # Product of all elements

print("25th Percentile (Q1):", np.percentile(arr, 25))  # First quartile

print("Covariance Matrix (self with self):", np.cov(arr))  # Covariance (needs 2D input usually)

print("Correlation Coefficient:", np.corrcoef(arr, arr2))  # Pearson correlation matrix

# --------------------------
# Broadcasting
# --------------------------
arr1 = np.array([1, 2, 3])  # 1D array
arr2 = np.array([[10], [20], [30]])  # 2D column vector
# Broadcasting: arr1 shape (3,) is broadcast across arr2 shape (3,1)
print("Broadcasting Addition:\n", arr1 + arr2)

# --------------------------
# Boolean indexing
# --------------------------
data = np.array([12, 7, 5, 19, 21, 8])  # Sample 1D array

# Boolean indexing: select elements greater than 10
print("Values > 10:", data[data > 10])

# Boolean indexing: select even numbers
print("Even numbers:", data[data % 2 == 0])

# Conditional replacement using np.where: replace elements < 10 with 0
print("Replace < 10 with 0:", np.where(data < 10, 0, data))

# --------------------------
# Set operations
# --------------------------
a = np.array([1, 2, 3, 4, 5])  # First array
b = np.array([3, 4, 5, 6, 7])  # Second array

# Union: all unique elements from both arrays
print("Union:", np.union1d(a, b))

# Intersection: elements common to both arrays
print("Intersection:", np.intersect1d(a, b))

# Difference: elements in a but not in b
print("Difference (a-b):", np.setdiff1d(a, b))

# Symmetric Difference: elements in a or b but not both
print("Symmetric Difference:", np.setxor1d(a, b))

# --------------------------
# Sorting & partitioning
# --------------------------
arr3 = np.array([5, 2, 8, 1, 7])  # Original array

# Sorted array in ascending order
print("Sorted:", np.sort(arr3))

# Indices that would sort the array
print("Argsort:", np.argsort(arr3))

# Partition array so that the two smallest elements are at the beginning
print("Partition (smallest 2):", np.partition(arr3, 2))

# --------------------------
# Linear Algebra basics
# --------------------------
mat1 = np.array([[1, 2], [3, 4]])  # First 2x2 matrix
mat2 = np.array([[5, 6], [7, 8]])  # Second 2x2 matrix

# Dot product of two matrices
print("Dot Product:\n", np.dot(mat1, mat2))

# Matrix multiplication (same as dot for 2D)
print("Matrix Multiplication:\n", np.matmul(mat1, mat2))

# Transpose of the first matrix
print("Transpose:\n", mat1.T)

# Determinant of the first matrix
print("Determinant:", np.linalg.det(mat1))

# Inverse of the first matrix
print("Inverse:\n", np.linalg.inv(mat1))

# Eigenvalues of the first matrix
print("Eigenvalues:", np.linalg.eig(mat1)[0])

# Eigenvectors of the first matrix
print("Eigenvectors:\n", np.linalg.eig(mat1)[1])

# --------------------------
# Random sampling
# --------------------------
# Randomly choose 3 unique elements from the list
print("Choice:", np.random.choice([1, 2, 3, 4, 5], size=3, replace=False))

# Random permutation of arr3
print("Permutation:", np.random.permutation(arr3))

# Generate 5 random numbers from standard normal distribution
print("Normal Distribution:", np.random.normal(0, 1, 5))

# --------------------------
# Array manipulations
# --------------------------
big = np.arange(24).reshape(4, 6)  # Create 4x6 array with numbers 0-23
print("Original:\n", big)

print("Transpose:\n", big.T)  # Transpose rows and columns

print("Ravel (1D view):", big.ravel())  # Flatten array to 1D view (no copy)

print("Split horizontally:", np.hsplit(big, 2))  # Split into 2 sub-arrays horizontally

print("Split vertically:", np.vsplit(big, 2))  # Split into 2 sub-arrays vertically

print("Repeat:", np.repeat([1, 2, 3], 3))  # Repeat each element 3 times

print("Tile:", np.tile([1, 2, 3], 3))  # Repeat the whole array 3 times

# --------------------------
# Math functions
# --------------------------
x = np.linspace(0, np.pi, 5)  # 5 evenly spaced numbers from 0 to π

print("Sine:", np.sin(x))  # Sine of each element

print("Log:", np.log(np.array([1, 10, 100])))  # Natural logarithm of array elements

print("Exp:", np.exp([1, 2, 3]))  # Exponential (e^x) of array elements

# --------------------------
# Broadcasting again
# --------------------------
a = np.arange(6).reshape(2, 3)  # 2x3 array: [[0,1,2],[3,4,5]]
b = np.array([10, 20, 30])      # 1D array to broadcast across rows

print("Broadcasted Add:\n", a + b)  # Element-wise addition using broadcasting

# --------------------------
# Masked arrays
# --------------------------
arr = np.array([1, 2, 3, -999, 5])  # Original array with invalid value -999
masked = np.ma.masked_equal(arr, -999)  # Mask all elements equal to -999

print("Masked Array:", masked)  # Show array with masked value
print("Masked Mean:", masked.mean())  # Compute mean ignoring masked value

# --------------------------
# Fancy indexing
# --------------------------
x = np.arange(10, 20)  # Array from 10 to 19
indices = [1, 3, 5, 7]  # Indices we want to access

print("Fancy Indexing:", x[indices])  # Access elements at specific indices

# --------------------------
# More Linear Algebra
# --------------------------
M = np.array([[1, 2], [3, 4]])  # 2x2 matrix

print("Trace:", np.trace(M))  # Sum of diagonal elements
print("Rank:", np.linalg.matrix_rank(M))  # Matrix rank

A = np.array([[1, 2], [3, 4]])  # Matrix A
B = np.array([[0, 5], [6, 7]])  # Matrix B

print("Kronecker Product:\n", np.kron(A, B))  # Kronecker product of A and B

block = np.block([[A, B], [B, A]])  # Block matrix concatenation
print("Block Matrix:\n", block)

# --------------------------
# Meshgrid example
# --------------------------
x = np.linspace(-1, 1, 5)  # 5 points from -1 to 1 for X-axis
y = np.linspace(-1, 1, 5)  # 5 points from -1 to 1 for Y-axis
X, Y = np.meshgrid(x, y)   # Create coordinate matrices from vectors
Z = np.sqrt(X**2 + Y**2)   # Compute radial distance from origin
print("Meshgrid X:\n", X)  # X coordinates grid
print("Meshgrid Y:\n", Y)  # Y coordinates grid
print("Radial Distance Z:\n", Z)  # Distance of each point from origin

# --------------------------
# FFT (Fast Fourier Transform)
# --------------------------
signal = np.array([1, 2, 3, 4])
fft_vals = np.fft.fft(signal)  # Compute FFT
print("FFT:", fft_vals)
print("Inverse FFT:", np.fft.ifft(fft_vals))  # Inverse FFT

# --------------------------
# Polynomial operations
# --------------------------
p = np.poly1d([1, -2, 1])  # Define polynomial x^2 - 2x + 1
print("Polynomial:", p)
print("Roots:", p.roots)  # Roots of polynomial
print("Derivative:", p.deriv())  # Derivative
print("Integral:", p.integ())  # Integral

# --------------------------
# Multivariate normal distribution
# --------------------------
print("Multivariate Normal:\n", np.random.multivariate_normal(
    mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=5))

# --------------------------
# Memory usage
# --------------------------
big_array = np.arange(1e6, dtype=np.float32)  # Large array
print("Memory (MB):", big_array.nbytes / 1e6)

# --------------------------
# Outer product
# --------------------------
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
print("Outer Product:\n", np.outer(u, v))

# --------------------------
# Sliding windows
# --------------------------
arr = np.arange(10)
print("Sliding Windows:\n", sliding_window_view(arr, window_shape=3))

# --------------------------
# Vectorize functions
# --------------------------
def my_func(x, y):
    return x**2 + y**2

vec_func = np.vectorize(my_func)
print("Vectorized Func:\n", vec_func([1, 2, 3], [4, 5, 6]))

# --------------------------
# Handling NaN values
# --------------------------
data = np.array([1, 2, np.nan, 4, 5])
print("Original Data:", data)
print("Mean ignoring NaN:", np.nanmean(data))
print("Sum ignoring NaN:", np.nansum(data))

masked_data = np.ma.masked_where(data < 3, data)  # Mask values < 3
print("Masked Data:", masked_data)
print("Masked Mean:", masked_data.mean())

# --------------------------
# Boolean indexing
# --------------------------
arr = np.arange(20).reshape(4, 5)
print("Original Array:\n", arr)
print("Elements >10:\n", arr[arr > 10])

rows = [0, 2]
cols = [1, 3]
print("Selected elements:", arr[rows, cols])

# --------------------------
# Rolling window / Moving average
# --------------------------
time_series = np.arange(10)
windowed = sliding_window_view(time_series, window_shape=3)
print("Rolling Window View:\n", windowed)
moving_avg = windowed.mean(axis=1)
print("Moving Average:", moving_avg)

# --------------------------
# Weighted sum
# --------------------------
features = np.array([[1, 2], [3, 4], [5, 6]])
weights = np.array([0.1, 0.9])
weighted_sum = features * weights
print("Weighted Features:\n", weighted_sum)
print("Weighted Sum across features:", weighted_sum.sum(axis=1))

# --------------------------
# One-hot encoding
# --------------------------
labels = np.array([0, 2, 1, 3])
num_classes = 4
one_hot = np.eye(num_classes)[labels]  # One-hot encode labels
print("One-hot encoded:\n", one_hot)

# --------------------------
# Covariance & Correlation
# --------------------------
X = np.random.rand(5, 4)
cov_matrix = np.cov(X, rowvar=False)
corr_matrix = np.corrcoef(X, rowvar=False)
print("Covariance Matrix:\n", cov_matrix)
print("Correlation Matrix:\n", corr_matrix)

# --------------------------
# Linear Regression (Normal Equation)
# --------------------------
np.random.seed(0)
X = np.random.rand(100, 2)
y = 2*X[:,0] + 3*X[:,1] + np.random.randn(100)*0.1
X_bias = np.c_[np.ones(X.shape[0]), X]  # Add bias column
w = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
print("Regression Coefficients:", w)

# --------------------------
# Multinomial & Dirichlet
# --------------------------
prob = [0.1, 0.2, 0.3, 0.4]
samples = np.random.multinomial(n=10, pvals=prob)
print("Multinomial Samples:", samples)

alpha = [0.5, 1.5, 2.0]
dir_samples = np.random.dirichlet(alpha, size=5)
print("Dirichlet Samples:\n", dir_samples)

# --------------------------
# PCA example
# --------------------------
data = np.random.rand(10, 3)
data_centered = data - data.mean(axis=0)
cov = np.cov(data_centered, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
projected = data_centered @ eigvecs[:, :2]  # Project onto top 2 PCs
print("Projected Data (PCA 2D):\n", projected)

# --------------------------
# Histogram
# --------------------------
data = np.random.randn(1000)
bins = np.linspace(-3, 3, 10)
hist, bin_edges = np.histogram(data, bins=bins)
print("Histogram counts:", hist)
print("Bin edges:", bin_edges)

# --------------------------
# Memory efficiency of views
# --------------------------
big_array = np.arange(1000000)
view_array = big_array[::2]
print("Original size:", big_array.nbytes/1e6, "MB")
print("View size (no extra memory):", view_array.nbytes/1e6, "MB")

# --------------------------
# Sliding windows on matrix
# --------------------------
mat = np.arange(16).reshape(4,4)
windows = sliding_window_view(mat, (2,2))
print("2x2 Windows:\n", windows)

# --------------------------
# Vectorized function example
# --------------------------
def f(x, y):
    return np.sin(x) * np.exp(y)

X = np.linspace(0, np.pi, 5)
Y = np.linspace(0, 1, 5)
result = f(X[:, None], Y[None, :])  # Vectorized computation on grids
print("Vectorized Result:\n", result)