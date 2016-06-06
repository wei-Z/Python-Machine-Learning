# Chapter 7 Combining Different Models for Ensemble Learning
# Learning with ensembles
from scipy.misc import comb
import math
def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probs = [comb(n_classifier, k) * error**k * (1- error) ** (n_classifier - k)
             for k in range(int(k_start), n_classifier + 1)]
    return sum(probs)

ensemble_error(n_classifier=11, error=0.25)
'''
After we've implemented the ensemble_error function, we can compute
the ensemble error rates for a range of different base errors from 0.0 to 1.0
to visualize the relationship between ensemble and base errors in a line graph:
'''
import numpy as np
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]

import matplotlib.pyplot as plt
plt.plot(error_range, ens_errors, label='Ensemble error', linewidth=2)
plt.plot(error_range, error_range, linestyle='--', label='Base error', linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid()
plt.show()

'''
As we can see in the resulting plot, the error probability of an ensemble is always
better than the error of an individual base classifier as long as the base classifiers
perform better than random guessing ( ε < 0.5 ). Note that the y-axis depicts the
base error (dotted line) as well as the ensemble error (continuous line):
'''

# Implementing a simple majority vote classifier
'''
To translate the concept of the weighted majority vote into Python code, we can use
NumPy's convenient argmax and bincount functions:'''
import numpy as np
np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6]))

'''
To implement the weighted majority vote based on class probabilities, we can again
make use of NumPy using numpy.average and np.argmax'''
ex = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
p
np.argmax(p)

'''Putting everythin together, let's now implement a MajorityVoteClassifier in Python'''
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    '''A majority vote ensemble classifier
    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble
    vote : str, {'classlabel', 'probability'}
      Default: 'classlabel'
      If 'classlabel' the prediction is based on
      the argmax of class labels. Else if
      'probability', the argmax of the sum of
      probabilities is used to predict the class label
      (recommended for calibrated classifiers).
    weights : array-like, shape = [n_classifiers]
      Optional, default: None
      If a list of `int` or `float` values are
      provided, the classifiers are weighted by
      importance; Uses uniform weights if `weights=None`. '''
    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}

        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        '''Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix},
            shape = [n_samples, n_features]
            Matrix of training samples.
        y : array-like, shape = [n_samples]
            Vector of target class labels.
        Returns
        -------
        self : object'''
        # Use LabelEncoder to ensure class labels start
        # with 0, which is important for np.argmax
        # call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = [ ]
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        '''Predict class labels for X.
        Parameters
       ----------
       X : {array-like, sparse matrix},
           Shape = [n_samples, n_features]
           Matrix of training samples.
       Returns
       ----------
       maj_vote : array-like, shape = [n_samples]
           Predicted class labels.'''
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)

        else: # 'classlabel' vote
            # Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        '''Predict class probabilities for X.
        Parameters
        ----------
        X : {array-like, sparse matrix},
            shape = [n_samples, n_features]
            Training vectors, where n_samples is
            the number of samples and
            n_features is the number of features.
        Returns
        ----------
        avg_proba : array-like,
            shape = [n_samples, n_classes]
            Weighted average probability for
            each class per sample.'''
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        '''Get classifier parameter names for GridSearch'''
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in\
                six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
        '''
        Also, note that we defined our own modified version of the get_params methods
        to use the _name_estimators function in order to access the parameters of individual
        classifiers in the ensemble. This may look a little bit complicated at first, but it will
        make perfect sense when we use grid search for hyperparameter-tuning in later sections.'''

# Combining different algorithms for classification with majority vote

'''We will take a shortcut and load the Iris dataset from scikit-learn's dataset module.
Furthermore, we will only select two features, sepal width and petal length, to make
the classification task more challenging. Although our MajorityVoteClassifier generalizes
to multiclass problems, we will only classify flower samples from the two classes,
Iris-Versicolor and Iris-Virginica, to compute the ROC AUC. The code is as follows:'''

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
'''
Next we split the Iris samples into 50 percent training and 50 percent test data:'''
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.5, random_state=1)
'''
Using the training dataset, we now will train three different classifiers—a
logistic regression classifier, a decision tree classifier, and a k-nearest neighbors
classifier—and look at their individual performances via a 10-fold cross-validation
on the training dataset before we combine them into an ensemble classifier:'''
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
print '10- fold cross validation:\n'

for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print 'ROC AUC: %0.2f (+/- %0.2f) [%s]' % (scores.mean(), scores.std(), label)

'''
Now let's move on to the more exciting part and combine the individual classifiers
for majority rule voting in our MajorityVoteClassifier:'''
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print 'Accuracy: %.2f (+/- %.2f) [%s]' % (scores.mean(), scores.std(), label)
'''
As we can see, the performance of the MajorityVotingClassifier has substantially improved
over the individual classifiers in the 10-fold cross-validation evaluation.'''

# Evaluating and tuning the ensemble classifier
'''
the MajorityVoteClassifier generalizes well to unseen data. We should remember that
the test set is not to be used for model selection; its only purpose is to report an unbiased
estimate of the generalization performance of a classifier system. The code is as follows:'''
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls \
    in zip(all_clf, clf_labels, colors, linestyles):
    # assuming the label of the positive class is 1
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
'''
As we can see in the resulting ROC, the ensemble classifier also performs well on the
test set (ROC AUC = 0.95), whereas the k-nearest neighbors classifier seems to be
overfitting the training data (training ROC AUC = 0.93, test ROC AUC = 0.86):'''
'''
Since we only selected two features for the classification examples, it would be interesting
to see what the decision region of the ensemble classifier actually looks like. Although it is
not necessary to standardize the training features prior to model fitting because our logistic
regression and k-nearest neighbors pipelines will automatically take care of this, we will
standardize the training set so that the decision regions of the decision tree will be on the
same scale for visual purposes. The code is as follows:'''
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
from itertools import product
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                      np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2,
                         sharex='col',
                         sharey='row',
                         figsize=(7, 5))

for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contour(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
                                                  X_train_std[y_train==0, 1],
                                                  c='red',
                                                  marker='o',
                                                  s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0],
                                                  X_train_std[y_train == 1, 1],
                                                  c='red',
                                                  marker='o',
                                                  s=50)
    axarr[idx[0], idx[1]].set_title(tt)

plt.text(-3.5, -4.5, s='Sepal width [standardized]', ha='center', va='center', fontsize=12)
plt.text(-10.5, 4.5,
          s='Petal length [standardized]',
          ha='center', va='center',
          fontsize=12, rotation=90)
plt.show()

'''
Interestingly but also as expected, the decision regions of the ensemble classifier seem
to be a hybrid of the decision regions from the individual classifiers. At first glance, the
majority vote decision boundary looks a lot like the decision boundary of the k-nearest
neighbor classifier. However, we can see that it is orthogonal to the y axis for sepal width
 ≥1, just like the decision tree stump:'''
'''
Before you learn how to tune the individual classifier parameters for ensemble classification,
let's call the get_params method to get a basic idea of how we can access the individual
parameters inside a GridSearch object:'''
mv_clf.get_params()

'''
Based on the values returned by the get_params method, we now know how to access the
individual classifier's attributes. Let's now tune the inverse regularization parameter C of the
logistic regression classifier and the decision tree depth via a grid search for demonstration
purposes. The code is as follows:
'''
from sklearn.grid_search import GridSearchCV
params = {'decisiontreeclassifier__max_depth': [1, 2], 'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

'''

After the grid search has completed, we can print the different hyperparameter value combinations
and the average ROC AUC scores computed via 10-fold cross-validation. The code is as follows:'''
for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f+/-%0.2f %r"
        % (mean_score, scores.std() / 2, params))

print 'Best parameters: %s' % grid.best_params_

print 'Accuracy: %.2f' % grid.best_score_

'''
As we can see, we get the best cross-validation results when we choose a lower regularization strength
(C = 100.0) whereas the tree depth does not seem to affect the performance at all, suggesting that a decision
stump is sufficient to separate the data. To remind ourselves that it is a bad practice to use the test dataset
more than once for model evaluation, we are not going to estimate the generalization performance of the tuned
hyperparameters in this section. We will move on swiftly to an alternative approach for ensemble learning: bagging.'''