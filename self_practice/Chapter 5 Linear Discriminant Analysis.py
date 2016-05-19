# Chapter 5 Compressing Data via Dimensionality Reduction
# Linear Discriminant Analysis(LDA)

# Supervised data compression via linear discriminant analysis
import numpy as np
import pandas as pd
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

 