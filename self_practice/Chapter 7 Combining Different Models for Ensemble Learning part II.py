# Bagging - building an ensemble of classifiers from bootstrap samples
'''
To see bagging in action, let's create a more complex classification problem using
the Wine dataset. Here, we will only consider the Wine classes 2 and 3, and we
select two features: Alcohol and Hue.'''
import numpy as np
import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
                                   'Malic acid', 'Ash',
                                   'Alcalinity of ash',
                                   'Magnesium', 'Total phenols',
                                   'Flavanoids', 'Nonflavanoid phenols',
                                   'Proanthocyanins',
                                   'Color intensity', 'Hue',
                                   'OD280/OD315 of diluted wines',
                                   'Proline']
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'Hue']].values

'''
Next we encode the class labels into binary format and split the dataset into
60 percent training and 40 percent test set, respectively:'''
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.40, random_state=1)

'''
A BaggingClassifier algorithm is already implemented in scikit-learn, which we
can import from ensemble submodule. Here, we will use an unpruned decision
tree as the base classifier and create an ensumble of 500 decision tree fitted on
different bootstrap samples of the training dataset:'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)
bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=1,
                        random_state=1)
'''
Next we will calculate the accuracy score of the prediction on the training and test
dataset to compare the performance of the bagging classifier to the performance of a
single unpruned decision tree:'''
from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print 'Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test)

'''Based on the accuracy values that we printed by executing the preceding
code snippet, the unpruned decision tree predicts all class labels of the training
samples correctly; however, the substantially lower test accuracy indicates high
variance (overfitting) of the model:'''
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print 'Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test)

'''
Although the training accuracies of the decision tree and bagging classifier are similar
on the training set (both 1.0), we can see that the bagging classifier has a slightly better
generalization performance as estimated on the test set. Next let's compare the decision
regions between the decision tree and bagging classifier:'''
import matplotlib.pyplot as plt

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 3))
for idx, clf, tt in zip([0, 1], [tree, bag], ['Decision Tree', 'Bagging']):
    clf.fit(X_train, y_train)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],
                                  X_train[y_train==0, 1],
                                  c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train == 1, 0],
                                   X_train[y_train == 1, 1],
                                   c='red', marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(10.2, -1.2, s='Hue', ha='center', va='center', fontsize=12)
plt.show()
'''
We only looked at a very simple bagging example in this section. In practice, more
complex classification tasks and datasets' high dimensionality can easily lead to
overfitting in single decision trees and this is where the bagging algorithm can really
play out its strengths. Finally, we shall note that the bagging algorithm can be an
effective approach to reduce the variance of a model. However, bagging is ineffective
in reducing model bias, which is why we want to choose an ensemble of classifiers
with low bias, for example, unpruned decision tree. '''
