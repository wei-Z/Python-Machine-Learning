import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder # change to numerical data
from sklearn.preprocessing import Imputer # fill in NA data
imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

df = pd.read_csv("/Users/Wei/Desktop/Python-Machine-Learning/Kaggle/train2016.csv")
le = LabelEncoder()

X_train = df.drop(['USER_ID', 'Party'], axis=1)
y_train = df['Party']
#########done reading train raw data###########
df1 = pd.read_csv("/Users/Wei/Desktop/Python-Machine-Learning/Kaggle/test2016.csv")
X_test = df1.drop(['USER_ID'], axis=1)
#########done reading test raw data###########

# Change all string to numerical values
X_train = X_train.apply(lambda x:x.fillna(x.value_counts().index[0]))
X_test = X_test.apply(lambda x:x.fillna(x.value_counts().index[0]))

for col in X_train.columns:
    le.fit(X_train[col])
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)

# Do I need to LabelEncoder y?

# logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# svm model
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train, y_train)

# decision tree model
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)

# randomforest model
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)



# Check the accuracy of training dataset
from sklearn.metrics import accuracy_score

# for logistic regression
y_train_pred = lr.predict(X_train)  
print accuracy_score(y_true=y_train, y_pred=y_train_pred)
# for svm
y_train_pred = svm.predict(X_train)
print accuracy_score(y_true=y_train, y_pred=y_train_pred)
# for decision tree
y_train_pred = tree.predict(X_train)
print accuracy_score(y_true=y_train, y_pred=y_train_pred)
# for random forest 
y_train_pred = forest.predict(X_train)
print accuracy_score(y_true=y_train, y_pred=y_train_pred)


# Predict test dataset
lr.predict(X_test)
svm.predict(X_test)
tree.predict(X_test)
forest.predict(X_test)

result= pd.concat([df1['USER_ID'], pd.Series(forest.predict(X_test))], axis=1)
result.columns=['USER_ID', 'Predictions']

result.to_csv('/Users/Wei/Desktop/Python-Machine-Learning/Kaggle/result4.csv', header=True, index=False)

