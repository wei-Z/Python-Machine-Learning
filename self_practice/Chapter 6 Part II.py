# Loading the Breast Cancer Wisconsin dataset
import pandas as pd
import numpy as np
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

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,
                           'clf__kernel': ['linear']},
                         {'clf__C': param_range,
                          'clf__gamma': param_range,
                          'clf__kernel': ['rbf']}]


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

# Looking at different performance evaluation metrics
# Reading a confusion matrix
from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print confmat

'''
We can map onto the confusion matrix illustration in the previous figure using
matplotlib's matshow function.
'''
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

'''
The scoring metrics are all implemented in scikit-learn and can be imported from
the sklearn.metrics module, as shown in the following snippet'''
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
print 'Precision: %3f' % precision_score(y_true=y_test, y_pred=y_pred)
print 'Precision: %3f' % recall_score(y_true=y_test, y_pred=y_pred)
print 'Precision: %3f' % f1_score(y_true=y_test, y_pred=y_pred)

from sklearn.metrics import make_scorer, f1_score
scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring=scorer, cv=10)

# Plotting a receiver operating characteristic(ROC curve)
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
pipe_lr = Pipeline([('scl', StandardScaler()),
                                 ('pca', PCA(n_components=2)),
                                 ('clf', LogisticRegression(penalty='l2', random_state=0, C=100.0))])
X_train2 = X_train[:, [4,14]]
cv = StratifiedKFold(y_train, n_folds=3, random_state=1)
fig = plt.figure(figsize=(7,5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = [ ]
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

plt.plot([0, 1],
            [0, 1],
            linestyle='--',
            color=(0.6, 0.6, 0.6),
            label='random guessing')
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
             label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1],
            [0, 1, 1],
            lw=2,
            linestyle=':',
            color='black',
            label='perfect performance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")
plt.show()

'''
If we are just interested in ROC AUC score, we could also directly import the 
roc_auc_score function from the sklearn.metrics submodule. The following code
calculates the classifier's ROC AUC score on the independent test dataset after fitting 
it on the two-feature training set: '''
pipe_lr = pipe_lr.fit(X_train2, y_train)
y_pred2 = pipe_lr.predict(X_test[:, [4,14]])

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
print 'ROC AUC: %.3f' % roc_auc_score(y_true=y_test, y_score=y_pred2)
print 'Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred2)

# The scoring metrics for multiclass classification
from sklearn.metrics import precision_score, make_scorer
pre_scorer = make_scorer(score_func=precision_score, pos_label=1, greater_is_better=True, average='micro')
