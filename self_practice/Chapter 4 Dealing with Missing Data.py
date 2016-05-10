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



