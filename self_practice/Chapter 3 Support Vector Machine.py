# -*- coding: utf-8 -*-
# Maximum margin classification with support vector machines
# Train a SVM model to classify the different flowers in our Iris dataset
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

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

######################Needed from previous section##############################
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# Alternative implementations in scikit-learn
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss='perceptron') # perceptron
lr = SGDClassifier(loss='log') # logistic regression
svm = SGDClassifier(loss='hinge') # support vector machine

# Solving nonlinear problems using a kernel SVM
np.random.seed(0) # called when RandomState is initialized. It can be called again to re-seed the generator
X_xor = np.random.randn(200, 2)
#Return a sample (or samples) from the “standard normal” distribution
# 200 rows, and 2 columns
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0) 
#Exclusive disjunction or exclusive or 
# is a logical operation that outputs true only when inputs differ (one is true, the other is false).
y_xor = np.where(y_xor, 1, -1) # if true, assign 1, if false, assign -1

plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

# Using the kernel trick to find separating hyperplanes in higher dimensional space
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

# Apply RBF kernel SVM to our Iris flower dataset
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# Increase the value of gamma, and observe the effect on the decision boundary:
svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# Risk of overfitting though

