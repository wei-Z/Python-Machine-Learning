# Predicting Continuous Target Variable with Regression Analysis
# Explore the Housing Dataset
import pandas as pd 
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', \
                       'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

# Visualizing the important characteristics of a dataset
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()

'''
In the following code, we will use Numpy's corrcoef function on the five feature columns that
we previously visualized in the scatterplot matrix, and we will use seaborn's heatmap function
to plot the correlation matrix array as a heat map.
'''
import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)
plt.show()

# Implementing an ordinary least squares linear regression model
# Solving regression for regression parameters with gradient descent
class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y-output)
            self.w_[1] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


'''
To see our LinearRegressionGD regressor in action, let's use the RM(number of rooms)
variable from the Housing Data Set as the explanatory variable to train a model that can
predict MEDV (the housing prices). Furthermore, we will standardize the variable for
better convergence of the GD algorithm. '''
X = df[['RM']].values
y =  df['MEDV'].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

'''
Now let's visualize how well the linear regression line fits the training data. To to
so, we will define a simple helper function that will plot a scatterplot of the training
samples and add the regression line:'''
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

'''
Now we will use this lin_regplot function to plot the number of rooms against
house pricces:'''
lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()

num_rooms_std = sc_x.transform([5.0])
price_std = lr.predict(num_rooms_std)
print "Price in $1000's: %.3f" % \
         sc_y.inverse_transform(price_std)

print 'Slope: %.3f' % lr.w_[1]
print 'Intercept: %.3f' % lr.w_[0]

# Estimating the coefficient of a regression model via scikit-learn
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
print 'Slope: %.3f' % slr.coef_[0]
print 'Intercept: %.3f' % slr.intercept_

lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()

# Fitting a robust regression model using RANSAC
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(),
                                                     max_trials=100,
                                                     min_samples=50,
                                                     residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                                                     residual_threshold=5.0,
                                                     random_state=0)
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()

print 'Slope: %.3f' % ransac.estimator_.coef_[0]
print 'Intercept: %.3f' % ransac.estimator_.intercept_

# Evaluating the performance of linear regression models
from sklearn.cross_validation import train_test_split
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

'''Using the following code, we will now plot a residual plot where we simply substract
the true target variables from our predicted responses:'''
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()

# MSE
from sklearn.metrics import mean_squared_error
print 'MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred))

#R^2
from sklearn.metrics import r2_score
print 'R^2 train: %3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred))

# Using regularized methods for regression
'''A Ridge Regression model can be initialized as follows:'''
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
'''Note that the regularization strength is regulated by the parameter alpha, which is 
similar to the parameter lambda. Likewise, we can initialize a LASSO regressor from 
the linear_model submodule:'''
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
'''Lastly, the ElasticNet implementation allows us to vary the L1 to L2 ratio'''
from sklearn.linear_model import ElasticNet
lasso = ElasticNet(alpha=1.0, l1_ratio=0.5)
'''For example, if we set l1_ratio to 1.0, the ElasticNet regressor would be equal to LASSO regression.'''

# Turning a linear regression model into a curve-polynomial regression
'''1. Add a second degree polynomial term:'''
from sklearn.preprocessing import PolynomialFeatures
X = np.array([258.0, 270.0, 294.0, 
              320.0, 342.0, 368.0, 
              396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8, 
              298.6, 314.2, 342.2, 
              360.8, 368.0, 391.2,
              390.8])
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

'''2. Fit a simple linear regression model for comparison:'''
lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

'''3. Fit a multiple regression model on the transformed feature for polynomial regression'''
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='quadratic fit')
plt.legend(loc='upper left')
plt.show()

y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
print 'Training MSE linear: %.3f, quadratic: %.3f' % (mean_squared_error(y, y_lin_pred), mean_squared_error(y, y_quad_pred))
print 'Training R^2 linear: %.3f, quadratic: %.3f' % (r2_score(y, y_lin_pred), r2_score(y, y_quad_pred))

# Modeling nonlinear relationships in the Housing Dataset
'''By executing the following code, we will model the relationship between house prices and LSTAT 
(percent lower status of the population) using second degree(quadratic) and third degree (cubic)
polynomials and compare it to a linear fit'''
X = df[['LSTAT']].values
y = df['MEDV'].values
regr = LinearRegression()

# create polynomial features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# linear fit
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

# quadratic fit
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

# cubic fit
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

# plot results
plt.scatter(X, y, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$)' % linear_r2, color='blue', lw=2, linestyle=':')
plt.plot(X_fit, y_quad_fit, label='quadratic (d=2), $R^2=%.2f$)' % quadratic_r2, color='red', lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit, label='cubic (d=3), $R^2=%.2f$)' % cubic_r2, color='green', lw=2, linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper right')
plt.show()

'''
In addition, polynomial features are not always the best choice for modeling nonlinear
relationships. For example, just by looking at the MEDV-LSTAT scatterplot, we could
propose that a log transformation of the LSTAT feature variable and the square root of
MEDV may project the data onto a linear feature space suitable for a linear regression
fit. Let's test this hypothesis by executing the following code:
'''
# transform features
X_log = np.log(X)
y_sqrt = np.sqrt(y)

# fit features
X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

# plot results
plt.scatter(X_log, y_sqrt,
                label='training points',
                color='lightgray')
plt.plot(X_fit, y_lin_fit,
            label='linear (d=1), $R^2=%.2f$' % linear_r2,
            color='blue',
            lw=2)
plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000\'s [MEDV]}$')
plt.legend(loc='lower left')
plt.show()

# Dealing with nonlinear relationships using random forests
# Decision tree regression
from sklearn.tree import DecisionTreeRegressor
X = df[['LSTAT']].values
y = df['MEDV'].values
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.show()

# Random forest regression
'''Now, let's use all the features in the Housing Dataset to fit a random forest
regression model on 60 percent of the samples and evaluate its performance
on the remaining 40 percent.'''
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print 'MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred))
print 'R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred))

'''Lastly, let's also take a look at the residuals of the prediction:'''
plt.scatter(y_train_pred,
                 y_train_pred - y_train,
                 c='black',
                 marker='o',
                 s=35,
                 alpha=0.5,
                 label='Training data')
plt.scatter(y_test_pred,
                 y_test_pred - y_test,
                 c='lightgreen',
                 marker='s',
                 s=35,
                 alpha=0.7,
                 label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()