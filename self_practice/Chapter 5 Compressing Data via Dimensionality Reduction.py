# Chapter 5 Compressing Data via Dimensionality Reduction

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



