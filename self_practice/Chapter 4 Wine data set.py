import pandas as pd
import numpy as np
# Partitioning a dataset in training and test sets
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns=['Class label', 'Alcohol',
                                 'Malic acid', 'Ash',
                                 'Alcalinity of ash', 'Magnesium',
                                 'Total phenols', 'Flavanoids',
                                 'Nonflavanoid phenols',
                                 'Proanthocyanins',
                                 'Color intensity', 'Hue',
                                 'OD280/OD315 of diluted wines',
                                 'Proline']
print 'Class labels', np.unique(df_wine['Class label'])

df_wine.head()

from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

# Bringing features onto the same scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
mms = MinMaxScaler()
stdsc = StandardScaler()
mms.fit_transform(x)
stdsc.fit_transform(x)

# There are two common approaches to bringing different features onto the same scale:
# normalization and standardization.
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# Selecting meaningful features
'''
A reason for overfitting is that our model is too complex for the given training data
and common colutions to reduce  the generalization error are listed as follows:
. Collect more training data
. Introduce a penalty for complexity via regularization
. Choose a simpler model with fewer parameters
. Reduce the dimensionality of the data

Common ways to reduce overfitting by regularization and dimensionality reduction
via feature selection.

'''

# Sparse solutions with L1 regularization
# For regularized models in scikit-learn that support L1 regularization, we can simply
# set the penalty parameter to 'l1' to yield the sparse solution.
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1')
'''
Applied to the standardized Wine data, the L1 regularized logistic regression would
yield the following sparse solution.
'''
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print 'Training accuracy: ', lr.score(X_train_std, y_train)
print 'Test accuracy: ', lr.score(X_test_std, y_test)

'''
Both training and test accuracies(both 98 percent) do not indicate any overfitting 
of our model. When we access the intercept terms via the lr.intercept_attribute, 
we can see that the array returns three values:
'''
lr.intercept_

lr.coef_

'''
We noticed that the weight vectors are sparse, which means that they only have a 
few non-zero entries. 
'''

'''
Lastly, let's plot the regularization path, which is the weight coefficients of the different
features fro different regularization strengths:
'''
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan',
                'magenta', 'yellow', 'black',
                'pink', 'lightgreen', 'lightblue',
                'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)
    
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()
''' 
The resulting plot provides us with further insights about the behavior of L1 regularization. 
As we can see, all features weights will be zero if we penalizethe model with a strong 
regularization parameter(C<0.1); C is the inverse of the regularization parameter lambda.
'''

# Sequential feature selection algorithms
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator) # copy
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test =  \
                    train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1] #the following calculations are for the situation where dim = self.k_features
        self.indices_ = tuple(range(dim)) # (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        self.subsets_ = [self.indices_] # [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            # for each value of dim, the index and score for larget score combination will be pushed to self.indices_,self.score_
            best = np.argmax(scores) # find the index with largest score
            self.indices_ = subsets[best] # find the combination index for the largest score
            self.subsets_.append(self.indices_) # push the index combination into self.subsets
            dim -= 1

            self.scores_.append(scores[best]) # push the largest score to self.scores_
        self.k_score_ = self.scores_[-1] # the score based on k_features 

        return self

    def transform(self, X):
        return X[:, self.indices]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices): # score calculation
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

'''
Let's see our SBS implementation in action using the KNN classification from scikit-learn
'''
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1) #number of features from 13 to 1 
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_] # we only care about how many features used during plot
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

'''
To satisfy our own curiosity, let's see what those five features are that yielded such a
good performance on the validation dataset:
'''
k5 = list(sbs.subsets_[8]) # index is 8 for this list [13,12,11,10,9,8,7,6,5...]
print df_wine.columns[1:][k5]

'''
Next let's evaluate the performance of the KNN classifier on the original test set
'''
knn.fit(X_train_std, y_train)
print 'Training accuracy: ', knn.score(X_train_std, y_train)
print 'Test accuracy: ', knn.score(X_test_std, y_test)

'''
Now let's use the selected 5-feature subset and see how well KNN performs
'''
knn.fit(X_train_std[:, k5], y_train)
print 'Training accuracy: ', knn.score(X_train_std[:, k5], y_train)
print 'Test accuracy: ', knn.score(X_test_std[:, k5], y_test)

# Accesing feature importance with random forests
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:] # 0 index is y column
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1 )
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1] # accending sort index, then change to decending index

for f in range(X_train.shape[1]):
    print "%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]])
    
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim(-1, X_train.shape[1])
plt.tight_layout()
plt.show()

X_selected = forest.transform(X_train, threshold=0.15)
X_selected.shape

#___________________________________________________________________#

# numpy.argsort(a, axis=-1, kind='quicksort', order=None)
'''
Returns the indices that would sort an array.
Perform an indirect sort along the given axis using the algorithm specified by the kind keyword. 
It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
'''

X_train.shape[1]
#Out[7]: 13L
X_train.shape
#Out[8]: (124L, 13L)
indices
#Out: array([ 9, 12,  6, 11,  0, 10,  5,  3,  1,  8,  4,  7,  2], dtype=int64)
importances[indices]
#Out: 
#array([ 0.18250763,  0.15857438,  0.15095391,  0.13198329,  0.10656371,
#        0.07824855,  0.06071706,  0.03203891,  0.02538503,  0.02236895,
#        0.02207032,  0.01465534,  0.01393292])


#itertools.combinations(iterable, r)
'''
Return r length subsequences of elements from the input iterable. 
So, if the input iterable is sorted, the combination tuples will be produced in sorted order.
'''
#numpy.argmax(a, axis=None, out=None)[source]
'''
Returns the indices of the maximum values along an axis.
'''