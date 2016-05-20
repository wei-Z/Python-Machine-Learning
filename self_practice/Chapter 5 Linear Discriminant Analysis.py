# Chapter 5 Compressing Data via Dimensionality Reduction
# Linear Discriminant Analysis(LDA)

# Supervised data compression via linear discriminant analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Computing the scatter matrices

np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
    print 'MV %s: %s\n' % (label, mean_vecs[label-1])
    
d = 13 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row-mv).dot((row-mv).T) # 13x1, 13x1 = 13x13
    S_W += class_scatter
print 'Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1])
print 'Class label distribution: %s' % (np.bincount(y_train)[1:])

# Using covariance 
d = 13 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print 'Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1])

mean_overall = np.mean(X_train_std, axis=0)
d = 13 # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train==i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print 'Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1])


# Selecting linear discriminants for the new feature subspace
'''
The remaining steps of the LDA are similar to the steps of the PCA. However,
instead of performing the eigendecomposition on the covariance matrix, we solve
the generalized eigenvalue problem of the matrix:
'''
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

'''
After we computed the eigenpairs, we can now sort the eigenvalues in
descending order:
'''
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print 'Eigenvalues in decreasing order: \n' 
for eigen_val in eigen_pairs:
    print eigen_val[0]
    
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1,14), discr, alpha=0.5, align='center', label='individual "discriminability"')
plt.step(range(1,14), cum_discr, where='mid', label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()

'''
Let's now stack the two most discriminative eigenvector columns to create the
transformation matrix W:
'''
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
print 'Matrix W: \n', w
 
# Projecting samples onto the new feature space
'''
Using the transformation matrix W that we created in the previous subsection,
we can now transform the training data set by multiplying the matrices:
'''
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0] * (-1), X_train_lda[y_train==l, 1] * (-1), c=c, label=1, marker=m)
    
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.show()

# LDA via scikit-learn
'''
The step-by-step implementation was a good exercise for understanding the inner
workings of LDA and understanding the differences between LDA and PCA. 
Now, let's take a look at the LDA class implemented in scikit-learn:
'''
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from Plot_Decision_Regions import plot_decision_regions
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

'''
Next, let's see how the logistic regression classifier handles the lower-dimensional 
training dataset after the LDA transformation:
'''
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()
'''
Looking at the resulting plot, we see that the logistic regression model misclassifies
one of the samples from class 2:
By lowering the regularization strength, we could probably shift the decision
boundaries so that the logistic regression models classify all samples in the training
dataset correctly. However, let's take a look at the results on the test set:
'''
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

'''
As we can see in the resulting plot, the logistic regression classifier is able to get a
perfect accuracy score for classifying the samples in the test dataset by only using a
two-dimensional feature subspace instead of the original 13 Wine features:
'''




