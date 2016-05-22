# Chapter 5 Compressing Data via Dimensionality Reduction
# Using kernel principal component analysis for nonlinear mappings
'''
Using kernel PCA, we will learn how to transform data that is not linearly spearable
onto a new, lower-dimensional subspace that is suitable for linear classifiers.'''

# Kernel functions and the kernel trick

# implementing a kernel principal component analysis in Python
'''
Now we are going to implement an RBF kernel PCA in Python following the three
steps that summaried the kernal PCA approach. Using the SciPy and Numpy helper
function's, we will see that implementing a kernel PCA is actually really simple.'''
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
def rbf_kernel_pca(X, gamma, n_components):

    '''
    RBF kernel PCA implementation.

    Parameters
    --------------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
        Tuning parameter of the RBF kernel

    n_components: int
        Number of principal components to return

    Returns
    -------------------
    X_pc: {Numpy ndarray}, shape = [n_samples, k_features]
        Projected dataset

    '''
    # Calculate pairwise squared Euclidean distance
    # in the MxN dimensional dataset
    sq_dists = pdist(X, 'sqeuclidean' )

    # Convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eivecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))

    return X_pc

# Example 1 - separating half-moon shapes
'''
Now, let's apply our rbf_kernel_pca on some nonlinear example datasets.
We will start by creating a two-dimensional dataset of 100 sample points
representing two half-moon shapes:'''
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.show()

'''
For the purposes of illustration, the half-moon of triangular symbols shall
represent one class and the half-moon depicted by the circular symbols
represent the samples from another class:'''
'''
Clearly, these two half-moon shapes are not linearly separable and our goal
is to unfold the half-moons via kernel PCA so that the dataset can serve as a
suitable input for a linear classifier. But first, let's see what the dataset looks
like if we project it onto the principal components via standard PCA:'''
from sklearn.decomposition import PCA
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1)),color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()