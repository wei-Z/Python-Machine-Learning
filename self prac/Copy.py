import numpy as np
class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1+ X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):  # range= [0:10]
            print _

            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0]  += update * 1

                errors += int(update != 0.0) # if update = 0.0, errors = 0; if update != 0.0, errors = 1;
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        # net input calculation
        return np.dot(X, self.w_[1:]) + self.w_[0] # matrix multiplication

    def predict(self, X):
        # return class label after unit step
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Traning a perception model on the Iris dataset
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
df.tail()

import matplotlib.pyplot as plt

y = df.iloc[0:100, 4].values # array([])
y = np.where(y == "Iris-setosa", -1, 1)

X = df.iloc[0:100, [0, 2]].values # array([])

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')

plt.show()

# Now it's time to train perceptron algorithm on this Iris
ppn = Perceptron(eta=0.1, n_iter=20)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

# ADAptive LInear NEuron (Adaline)
# Implementing an Adaptive Linear Neuron in Python


