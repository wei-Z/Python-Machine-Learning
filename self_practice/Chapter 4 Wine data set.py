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
ax = plt.subplot(lll)

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

    
