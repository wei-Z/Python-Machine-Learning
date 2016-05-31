# Loading the Breast Cancer Wisconsin dataset
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

le.transform(['M', 'B'])

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.20, random_state=1)

# Combining transformers and estimators in a pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, y_train)
print 'Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test)

# Algorithm selection with nested cross-validation
'''
In scikit-learn, we can perform nested cross-validation as follows:'''
gs = GridSearchCV(estimator=pipe_svc,
                                    param_grid=param_grid,
                                    scoring='accuracy',
                                    cv=2,
                                    n_jobs=-1)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print 'CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))

from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator = DecisionTreeClassifier(random_state=0),
                                    param_grid=[
                                        {'max_depth': [1,2,3,4,5,6,7,None]}],
                                    scoring='accuracy',
                                    cv=5)
scores = cross_val_score(gs,
                                              X_train,
                                              y_train,
                                              scoring='accuracy',
                                              cv=2)
print 'CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))
