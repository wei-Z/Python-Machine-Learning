# Dealing with missing data
import pandas as pd
from io import StringIO
csv_data = ''' A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
# If using Python 2.7, need
# to convert the string to unicode:
csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))
df

# Use the isnull method to return a DataFrame with Boolean values
# that indicate whether a cell contains a numeric value (False) or if
# data is missing (True).
# Use the sum method, we can then return the number of missing
# values per column as follows
df.isnull().sum()

# We can always access the underlying NumPy array of the DataFrame
# via the values attribute before we feed it into a scikit-learn estimator:
df.values

# Eliminating samples or features with missing values
''' One of the easiest ways to deal with missing data is to simply remove
the corresponding features (columns) or samples (rows) frome the dataset
entirely; rows with missing values can be easily dropped via the dropna method:'''
df.dropna()

''' Simplarily, we can drop columns that have at least one NaN in any
row by setting the axis argument to 1:'''
df.dropna(axis=1)

''' The dropna method supports several additional parameters that can
come in handy:'''
# only drop rows where all columns are NaN
df.dropna(how='all')
# drop rows that have not at least 4 non-NaN values
df.dropna(thresh=4)
# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])

# Imputing missing values
''' In this section we will look at one of the most commonly used alternatives
for dealing with missing values: interpolation techniques.
In this case, we can use different interpolation techniques to estimate the
missing values from the other training samples in our dataset.
One of the most common interpolation techniques is mean imputation,
where we simply replace the missing value by the mean value of  the
entire feature column.
A convenient way to achieve this is by using the Imputer class from
scikit-learn: '''
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
imputed_data
# We replaced each NaN value by the corresponding mean, which is separetely calculated for each 
# feature column. If we changed the setting axis=0 to axis=1, we'd calculate the row means.
# Other options for the strategy parameter are median or most_frequent, where the later replaces
# the missing values by the most frequent values.

# Handling categorical data
''' Create new data frame with categorical data '''
import pandas as pd
df = pd.DataFrame([
                             ['green', 'M', 10.1, 'class1'],
                             ['red', 'L', 13.5, 'class2'],
                             ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
df

# Mapping ordinal features
size_mapping = {
                        'XL': 3,
                        'L': 2,
                        'M': 1}

df['size'] = df['size'].map(size_mapping)
df
'''
We can simply define a reverse-mapping dictionary 
inv_size_mapping = {v: k for k, v in size_mapping.items()}

'''

# Encoding class labels
import numpy as np
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping

df['classlabel']= df['classlabel'].map(class_mapping)
df
'''
We can reverse the key-value pairs in the mapping directionary as follows to map the
converted class labels back to the original string representation:
    '''
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df

'''
Alternatively, there is a convenient LabelEncoder class directly implemented in scikit-learn
to achieve the same:

'''
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y

class_le.inverse_transform(y)

# Performing one-hot encoding on nominal features
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0]).astype(int)
X

from sklearn.preprocessing import OneHotEncoder

'''
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray() # TypeError: no supported conversion for types: (dtype('float64'), dtype('O')) 

'''
ohe = OneHotEncoder(categorical_features = [0], sparse=False) # This one works
ohe.fit_transform(X)

'''
An even more convenient way to create those dmmy features via one-hot encoding is to use 
the get_dummies method implemented in pandas. Applied on a DataFrame, the get_dummies 
method will only convert string columns and leave all other columns unchanged:

'''
pd.get_dummies(df[['price', 'color', 'size']])








