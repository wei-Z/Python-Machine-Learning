# Predicting Continuous Target Variable with Regression Analysis
# Explore the Housing Dataset
import pandas as pd 
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', \
                       'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

# Visualizing the important characteristics of a dataset
