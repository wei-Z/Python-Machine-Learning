import numpy as np
class Perceptron(object):
    """Perceptron classifier.
    
        Parameters
        ------------
        eta : float
        Learning rate (between 0.0 and 1.0)
        n_iter : int
        Passes over the training dataset.
        Attributes
        -----------
        w_ : 1d-array
        Weights after fitting.
        errors_ : list
        Number of misclassifications in every epoch.
        
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        """Fit training data.
        
            Parameters
            ----------
            X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples
            is the number of samples and
            n_features is the number of features.
            y : array-like, shape = [n_samples]
            Target values.
            Returns
            -----------
            self : object
            
        """
        self.w_ = np.zeros(1 + X.shape[1]) # return zero array, X.shape[1] =2L, return array([0., 0., 0.])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0) # if update = 0.0, errors = 0; if update unequal 0.0, errors =1. 
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        #"""Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0] # matrix multiplication
    def predict(self, X):
        #"""Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1) # numpy.where(condition[, x, y])
                                                                             # Return elements, either from x or y, depending on condition.



# Training a perceptron model on the Iris dataset
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

import matplotlib.pyplot as plt

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)# if y == 'Iris-setosa', y = -1, otherwise if y == 'Iris-versicolor', y =1.
X = df.iloc[0:100, [0,2]].values
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron()
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o', color="green")
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

# Implement a small convenience function to visualize the decision boundaries for 2D datasets:
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print "xx1: ", xx1
    print "xx2: ", xx2
    print "Z: ", Z
    print "xx1.ravel(): ", xx1.ravel()
    print "xx2.ravel(): ", xx2.ravel()
    Z = Z.reshape(xx1.shape)
    print "Z: ", Z
    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()
# meshgrid, ravel, reshape, contourf, xlim, ylim

# how to use contourf and meshgrid:
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x,y,z)      


        
    
