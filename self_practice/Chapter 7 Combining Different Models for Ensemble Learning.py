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
        self.lablenc.fit(y)
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
                    out['$s__%s' % (name, key)] = value
            return out
        '''
        Also, note that we defined our own modified version of the get_params methods
        to use the _name_estimators function in order to access the parameters of individual
        classifiers in the ensemble. This may look a little bit complicated at first, but it will
        make perfect sense when we use grid search for hyperparameter-tuning in later sections.'''


