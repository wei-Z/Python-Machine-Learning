# Chapter 5 Compressing Data via Dimensionality Reduction
# Principal Components Analysis
# Total and explained variance
'''
First, we will start by loading the Wine dataset that we have been working with
in Chapter 4'''
import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

'''
Next, we will process the Wine data into separate training and test sets using 70
percent and 30 percent of the data, respectively - and standardize it to unit variance'''
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

'''
We will use the linalg.eig function from Numpy to obtain the eigenpairs of the Wine
covariance matrix'''
import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print '\nEigenvalues \n%s' % eigen_vals
'''
Using the numpy.cov function, we computed the covariance matrix of the
standardized training dataset. Using the linalg.eig function, we performed the
eigendecomposition that yielded a vector (eigen_vals) consisting of 13 eigenvalues
 and the corresponding eigenvectors stored as columns in a 13x13 -dimensional
 matrix (eigen_vecs).
'''
'''
Since we want to reduce the dimensionality of our dataset by compressing it onto
a new feature subspace, we only select the subset of the eigenvectors(principle 
components) that contains most of the information (variance). Since the eigenvalues 
define the magitude of the eigenvectors, we have to sort the eigenvalues by 
decreasing magnitude; we are interested in the top k eigenvectors based on the 
values of their corresponding eigenvalues. 
But before we collect those k most informative eigenvectors, let's plot the variance
explained ratios of the eigenvalues

'''
'''
Using the NumPy cumsum function, we can then calculate the cumulative  sum of 
explained variances, which we will plot via matplotlib's step function: '''
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

'''
The resulting plot indicates that the first principal component alone accounts for 
40 percent of the variance. Also, we can see that the first two principal components
combined explain almost 60 percent of the variance in the data:
'''
# Feature transformation
# We start by sorting the eigenpairs by decreasing order of the eigenvalues:
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],   # eigen_pairs[0][1].shape =(13,)
                       eigen_pairs[1][1][:, np.newaxis]))  # eigen_pairs[0][1][:, np.newaxis].shape= (13, 1)
print 'Matrix W: \n', w

'''
By executing the preceding code, we have created a 13x2-dimensional projection
matrix w from the top two eigenvectors. Using the projection matrix, we can now
transform a sample x (represented as 1x13-dimensional row vector) onto the PCA
subspace obraining x', a now two-dimensional sample vector consisting of two new
features:       x' = xW
'''
X_train_std[0].dot(w) # 1x13 13x2 = 1x2
# similarly, we can transform the entire 124 x 13-dimensional training dataset onto the two principal components
# by calculating the matrix dot product:
X_train_pca = X_train_std.dot(w) # 124x13 13x2 = 124x2

'''
Lastly, let's visualize the transformed Wine training set, now stored as an 
124 x 2-dimensional matrix, in a two-dimensional scatterplot:
'''
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1],c=c, label=1, marker = m)
    
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

# Principal component analysis in scikit-learn
from Plot_Decision_Regions import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pca = PCA(n_components=2) # Number of components to keep. if n_components is not set all components are kept
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

'''
Let's plot the decision regions of the logistic regression on the transformed test 
dataset to see if it can separate the classes well.
'''
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

'''
If we are interested in the explained variance ratios of the different principal 
components, we can simply intialize the PCA class with the n_components parameter
set to None, so all principal components are kept and the explained variance ratio can 
then be accessed via the explained _variance_ratio_ attribute
'''
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_

'''
From Wikipedia:
The eigendecomposition of a symmetric positive semidefinite (PSD) matrix yields an 
orthogonal basis of eigenvectors, each of which has a nonnegative eigenvalue. The 
orthogonal decomposition of a PSD matrix is used in multivariate analysis, where the 
sample covariance matrices are PSD. This orthogonal decomposition is called principal 
components analysis (PCA) in statistics. PCA studies linear relations among variables. 
PCA is performed on the covariance matrix or the correlation matrix (in which each 
variable is scaled to have its sample variance equal to one). For the covariance or 
correlation matrix, the eigenvectors correspond to principal components and the 
eigenvalues to the variance explained by the principal components. Principal 
component analysis of the correlation matrix provides an orthonormal eigen-basis for 
the space of the observed data: In this basis, the largest eigenvalues correspond to 
the principal components that are associated with most of the covariability among a 
number of observed data.

'''




