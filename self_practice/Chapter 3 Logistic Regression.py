# -*- coding: cp1252 -*-
# Chapter 3
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target # 0-Iris-Setosa, 1-Iris-Versicolor, 2-Virginica

from sklearn.cross_validation import train_test_split
# random_state : int or RandomState
# Pseudo-random number generator state used for random sampling.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# everytime run it without specifying random_state, you will get a different result, this is expected behaviour.
# If you use random_state=some_number, then you can guarantee that your split will be always the same.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)  # only compute mean and std here
X_train_std = sc.transform(X_train) # perform standardization by centering and scaling
X_test_std = sc.transform(X_test) # perform standardization by centering and scaling

from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0) # Check source code of Perceptron
                                                                                                   # Why return void.
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print 'Misclassified samples: %d' % (y_test != y_pred).sum()


# calculate the classification accuracy of the perceptron on the test set as follows:
from sklearn.metrics import accuracy_score
print 'Accuracy: %.2f' % accuracy_score(y_test, y_pred)

# Plot the decision regions
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
# setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')

# Specify the indices of the samples that we want to mark on the resulting plots.
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
#####################Example of hstack and vstack##################
a = np.array([[1], [2], [3]])
b = np.array([[2], [3], [4]])
np.vstack((a,b))
#array([[1], [2], [3], [2], [3], [4]])
np.hstack((a,b))
#array([[1, 2], [2, 3], [3, 4]])
##############################################################

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
# The three flower classes can not be perfectly separated by a linear decision boundaries.

#############Modeling class probabilities via logistic regression###############
#--------------------------Feel odds ratio and logit function------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
p = np.arange(0.01,1,0.01)

def odds(x):
    return x/(1-x)
    
import matplotlib.pyplot as plt
plt.scatter(p, p/(1-p), c='g', marker='x')
plt.xlabel('p')
plt.ylabel('p/(1-p)')
plt.show()

def logit(x):
    return np.log(x/(1-x))
    
plt.scatter(p, logit(p), c='r', marker='s')
plt.xlabel('p')
plt.ylabel('logit function')
plt.show()

#-----------------------------------------------------------------------------------------------#
''' Plot the sigmoid function for some values in the range -7 to 7 to see what it looks like '''
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
    
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0, color='k')
# Add a vertical line across the axes.
# axvline(x=0, ymin=0, ymax=1, hold=None, **kwargs)

plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
# Add a horizontal span (rectangle) across the axis.
# axhspan(ymin, ymax, xmin=0, xmax=1, **kwargs)
# y coords are in data units and x coords are in axes(relative 0 - 1) units.
# ls or linestyle:
# ['solid' | 'dashed', 'dashdot', 'dotted' | (offset, on-off-dash-seq) | '-' | '--' | '-.' | ':' | 'None' | ' ' | '']
# facecolor or fc:
# mpl color spec, or None for default, or 'none' for no color

plt.axhline(y=0.5, ls='dotted', color='k')
# Add a horizontal line across the axis.
# axhline(y=0, xmin=0, xmax=1, hold=None, **kwargs)

plt.yticks([0.0, 0.5, 1.0])
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.show()
# We concluded that this sigmoid function take real number values as input and
# transform them to values in range [0,1] with an intercept at phi ( z ) = 0.5 .

#---------------------Training a logistic regression model with scikit-learn--------------------#
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
# defined in as above
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()


# Predict the class-membership probability of the samples via the predict_proba method.
# predict the probabilities of the first Iris-Setosa sample.
lr.predict_proba(X_test_std[0, :])

weights, params = [ ], [ ]
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
    
weights = np.array(weights)
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()

# As we can see in the resulting plot, the weight coefficients shrink if we decrease
# the parameter C, that is, if we increase the regularization strength.