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
    eigvals, eigvecs = eigh(K)

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

'''
Note that when we plotted the first principal component only (right subplot),
we shifted the triangular samples slightly upwards and the circular samples
slightly downwards to better visualize the class overlap'''

'''
Now let's try out our kernel PCA function rbf_kernel_pca, which we implemented
in the previous subsection:'''
from matplotlib.ticker import FormatStrFormatter
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((50, 1)) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50, 1)) - 0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()

'''
We can now see that the two classes (circles and triangles) are linearly well separated
so that it becomes a suitable training dataset for linear classifiers:'''
'''
Unfortuately, there is no universal value for the tuning parameter gamma that works
well for different datasets. To find a gamma value that is appropriate for a given problem
requires experimentation. '''

# Example 2 - separating concentric circles
'''
In the previous subsection, we showed how to separate half-moon shapes via
kernel-PCA. Since we put so much effort into understanding the concept of  kernel
PCA, let's take a look at another interesting example of a nonlinear problem:
concetric circles.'''
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.show()

'''Let's start with the standard PCA approach to compare it with the results of the RBF
kernel PCA:'''
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((500, 1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((500, 1))-0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()
''' Again, we can see that standard PCA is not able to produce results suitable for
training a linear classifier.'''
''' Given an appropriate value for gamma, let's see if we are luckier using the RBF
kernel PCA implementation:'''
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((500, 1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500, 1))-0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

# Projecting new data points
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    '''
    RBF kernel PCA implementation.

    Parameters
    -----------------
    X: {Numpy ndarray}, shape = {n_samples, n_features}

    gamma: float
        Tuning parameter of the RBF kernel

    n_components: int
        number of principal components to return

    Returns
    -----------------
    X_pc: {Numpy ndarray}, shape = [n_samples, k_features}
        Projected dataset

    lambdas: list
        Eigenvalues

    '''
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmertric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_components+1)))

    # Collect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_components+1)]

    return alphas, lambdas

'''
Now, let's create a new half-moon dataset and project it onto a one-dimensional
subspace using the updated RBF kernel PCA implementation:'''
X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)

'''
To make sure that we implement the code for projecting new samples, let's assume
that the 26th point from the half-moon dataset is a new data point x', and our task is
to project it onto this new subspace:'''
x_new = X[25]
x_new
x_proj = alphas[25] # original projection
x_proj

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

'''
By executing the following code, we are able to reproduce the original projection.
Using the project_x function, we will be able to project any new data samples as
well. The code is as follows:'''
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
x_reproj

'''
Lastly, let's visualize the projection on the first principal component.'''
plt.scatter(alphas[y==0, 0], np.zeros((50)), color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y==1, 0], np.zeros((50)), color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black', label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green', label='remapped point X[25]', marker='x', s=500)
plt.legend(scatterpoints=1)
plt.show()
'''
As we can see in the following scatterplot, we mapped the sample x' onto the first
principal component correctly:'''

# Kernel principal component analysis in scikit-learn
'''
For our convenience, scikit-learn implements a kernel PCA class in the
sklearn.decomposition submodule. The usage is similar to the standard
PCA class, and we can specify the kernel via the kernel parameter:'''
from sklearn.decomposition import KernelPCA
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca=scikit_kpca.fit_transform(X)
'''
To see if we get results that are consistent with our own kernel PCA
implementation, let's plot the transformed half-moon shape data onto the
first two principal components:'''
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.plt.show()

