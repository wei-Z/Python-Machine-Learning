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

from sklearn.precessing import StandardScaler
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
