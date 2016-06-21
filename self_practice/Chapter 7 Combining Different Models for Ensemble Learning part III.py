# Leveraging weak learners via adaptive boosting
'''
In this section about ensemble methods, we will discuss boosting with a special focus on
its most common implementation, AdaBoost(short for Adaptive Boosting).'''
'''
In boosting, the ensemble consists of very simple base classifiers, also often referred
to as weak learners, that have only a slight performance advantage over random
guessing. A typical example of a weak learner would be a decision tree stump.
The key concept behind boosting is to focus on training samples that are hard
to classify, that is, to let the weak learners subsequently learn from misclassified
training samples to improve the performance of the ensemble. In contrast to bagging,
the initial formulation of boosting, the algorithm uses random subsets of training
samples drawn from the training dataset without replacement. The original boosting
procedure is summarized in four key steps as follows:
'''
import pandas as pd
import numpy as np
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=0)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print 'Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test)
'''As we can see, the decision tree stump seems to overfit the training data in contrast
with the unpruned decision tree that we saw in the previous section: '''
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print 'AdaBoost train/test accuracies %.3f/%.3f' % (ada_train, ada_test)

'''
Although we used another simple example for demonstration purposes, we can
see that the performance of the AdaBoost classifier is slightly improved compared
to the decision stump and achieved very similar accuracy scores to the bagging
classifier that we trained in the previous section. However, we should note that it is
considered as bad practice to select a model based on the repeated usage of the test
set. The estimate of the generalization performance may be too optimistic, which we
discussed in more detail in Chapter 6, Learning Best Practices for Model Evaluation and
Hyperparameter Tuning.
Finally, let's check what the decision regions look like:'''
import matplotlib.pyplot as plt
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(1, 2,
sharex='col',
sharey='row',
figsize=(8, 3))
for idx, clf, tt in zip([0, 1], [tree, ada], ['Decision Tree', 'AdaBoost']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],
                              X_train[y_train==0, 1],
                              c='blue',
                              marker='^')
    axarr[idx].scatter(X_train[y_train==1, 0],
                              X_train[y_train==1, 1],
                              c='red',
                              marker='o')
axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(10.2, -1.2,
            s='Hue',
            ha='center',
            va='center',
            fontsize=12)
plt.show()
'''
By looking at the decision regions, we can see that the decision boundary of the
AdaBoost model is substantially more complex than the decision boundary of the
decision stump. In addition, we note that the AdaBoost model separates the feature
space very similarly to the bagging classifier that we trained in the previous section.'''
'''As concluding remarks about ensemble techniques, it is worth noting that
ensemble learning increases the computational complexity compared to individual
classifiers. In practice, we need to think carefully whether we want to pay the price
of increased computational costs for an often relatively modest improvement of
predictive performance.'''

