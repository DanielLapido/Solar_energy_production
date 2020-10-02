# MACHINE LEARNING MODELS FOR PREDICTING SOLAR ENERGY PRODUCTION
# AUTHOR: DANIEL LAPIDO MARTINEZ

# A few machine learning models (KNN, SVMs and REGRESSION TREES) that are able
# to estimate how much solar energy is going to be produced at one of the solar plants in Oklahoma.

#  The Global Forecast System (GFS, USA) gives a forecast everyday at 00:00 UTC for next day, at 5 times: 1 (12h), 2
# (15h), 3 (18h), 4 (21h), 5 (24h), for 15 meteorological variables at 16 locations in Oklahoma.

# At the solar plant of interest, the meteorological variables at the closest location
# are used to build a model that allows to predict the solar energy production.

#  The train dataset contains data from 1994-2007 (one day per row)
# and the test dataset, for 2008-2009.  The train set will be divided into train (10 first years)
# and validation (the last two years). This last one is used for hyper-parameter tuning.

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import PredefinedSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from scipy.stats import randint as sp_randint
from sklearn.utils.fixes import loguniform

# 1)
# Set seed for reproducibility
np.random.seed(123)

# 2)
#  Read the solar dataset (in pickle format) into a Pandas dataframe
train = pd.read_pickle('trainst1ns16.pkl')
test = pd.read_pickle('testst1ns16.pkl')

# 3)
# Normalize attributes to the range 0-1 using the MinMaxScaler


# All pre-processing is done with information obtained from the training partition (.fit),
# and then applied to the testing partition (.transform).
min_max_scaler = preprocessing.MinMaxScaler()
train_minmax = min_max_scaler.fit_transform(train)
test_minmax = min_max_scaler.transform(test)


# 4)
# Divide train set into train(first 10 years) and validation (last 2 years)

# Weather conditions might be similar between nearby days, so we do not want to shuffle the data
# If one of those days go to the train set and the other goes to the validation set, it will be biased

# Defining a fixed train/validation grid-search
# -1 means training, 0 means validation
# First 10 years (10*365 days) for validation:
validation_indices = np.zeros(train_minmax.shape[0])
validation_indices[:10*365] = -1
tr_val_partition = PredefinedSplit(validation_indices)

# 5)
# We are going to use only the closest location to the corresponding solar plant.
# This can be achieved by selecting the first 75 columns

train_closest = train_minmax[:, 0:75]
test_closest = test_minmax[:, 0:75]

# 6) Train (using the training data, which has the training and validation partitions) and evaluate (using the test set)

# a. KNN,  Regression Trees, and SVMs with default hyper-parameters


# ---------- DEFAULT HYPERPARAMETERS ----------

print("DEFAULT HYPERPARAMETERS")
print('-------------------------------------------------')

# KNN

# Hyperparameters:
# n_neighbors : (default = 5) Number of neighbors to use by default for :meth:`kneighbors` queries.
# There are other hyperparameters like the power parameter for the Minkowski metric or the weights used in prediction.
# However we will only later tune the n_neighbors hyperparameter.

# Defining the method
knn = KNeighborsRegressor()

# Training the model with reproducibility
np.random.seed(123)
knn.fit(train_closest, train_minmax[:, 1200])

# Making predictions on the testing partition
y_test_pred = knn.predict(test_closest)

# And finally computing the test accuracy
print("Mean squared error of KNN with default hyperparameters:",
      metrics.mean_squared_error(y_test_pred, test_minmax[:, 1200]))
print("R-squared:", metrics.r2_score(y_test_pred, test_minmax[:, 1200]))
print("-----------------------------------------------------------")

# REGRESSION TREES

# Hyperparameters:
# - max_depth :(default=None) The maximum depth of the tree. If None, then nodes are expanded until all leaves
#  are pure or until all leaves contain less than min_samples_split samples.

# - min_samples_split : (default=2) The minimum number of samples required to split an internal node.

# (There are more hyper-parameters, but these are the most important)

# Defining the method
clf = tree.DecisionTreeRegressor()

# Training the model with reproducibility
np.random.seed(123)
clf.fit(train_closest, train_minmax[:, 1200])

# Making predictions on the testing partition
y_test_pred = clf.predict(test_closest)

# And finally computing the test accuracy
print("Mean squared error of Regression trees with default hyperparameters:",
      metrics.mean_squared_error(y_test_pred, test_minmax[:, 1200]))
print("R-squared:", metrics.r2_score(y_test_pred, test_minmax[:, 1200]))
print("-----------------------------------------------------------")

# SUPPORT VECTOR MACHINES

# Hyperparameters:
# - kernel (default=’rbf’)Specifies the kernel type to be used in the algorithm.
# It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable

# - gamma (default=’scale’) Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

# - C (default=1.0) Regularization parameter. The strength of the regularization is inversely proportional to C.
# Must be strictly positive. The penalty is a squared L2 penalty.

# Defining the method
svr = SVR()

# Training the model with reproducibility
np.random.seed(123)
svr.fit(train_closest, train_minmax[:, 1200])

# Making predictions on the testing partition
y_test_pred = svr.predict(test_closest)

# And finally computing the test accuracy
print("Mean squared error of SVM with default hyperparameters:",
      metrics.mean_squared_error(y_test_pred, test_minmax[:, 1200]))
print("R-squared:", metrics.r2_score(y_test_pred, test_minmax[:, 1200]))
print("-----------------------------------------------------------")
print("-----------------------------------------------------------")

# ----------  HYPERPARAMETERS TUNING WITH RANDOM-SEARCH ------------

print("TUNED HYPERPARAMETERS WITH RANDOM-SEARCH")
print('-------------------------------------------------')

# KNN

param_dist = {'n_neighbors': sp_randint(2, 20)}

knnrs = RandomizedSearchCV(knn, param_distributions=param_dist, scoring='neg_mean_squared_error',
                           cv=tr_val_partition, n_jobs=1, verbose=1)

# Training the model with the random-search
np.random.seed(123)
knnrs.fit(train_closest, train_minmax[:, 1200])

# Making predictions on the testing partition
y_test_pred = knnrs.predict(test_closest)

# And finally computing the test accuracy
print("Mean squared error of KNN with tuned hyperparameters:",
      metrics.mean_squared_error(y_test_pred, test_minmax[:, 1200]))
print("R-squared:", metrics.r2_score(y_test_pred, test_minmax[:, 1200]))
print("-----------------------------------------------------------")

# REGRESSION TREES

param_dist = {'max_depth': sp_randint(2, 16),
              'min_samples_split': sp_randint(2, 34)}

n_iter_search = 20

clfrs = RandomizedSearchCV(clf, param_distributions=param_dist, scoring='neg_mean_squared_error',
                           cv=tr_val_partition, n_jobs=1, verbose=1, n_iter=n_iter_search)

# Training the model with the random-search
np.random.seed(123)
clfrs.fit(train_closest, train_minmax[:, 1200])

# Making predictions on the testing partition
y_test_pred = clfrs.predict(test_closest)

# And finally computing the test accuracy
print("Mean squared error of Regression trees with tuned hyperparameters:",
      metrics.mean_squared_error(y_test_pred, test_minmax[:, 1200]))
print("R-squared:", metrics.r2_score(y_test_pred, test_minmax[:, 1200]))
print("-----------------------------------------------------------")

# SUPPORT VECTOR MACHINES

# For continuous parameters, such as C , it is important to specify a continuous distribution
# to take full advantage of the randomization. A continuous log-uniform random variable is available through loguniform.

param_dist = {'C': loguniform(1e0, 1e3), 'gamma': loguniform(1e-4, 1e-3), 'kernel': ['rbf']}

svrrs = RandomizedSearchCV(svr, param_distributions=param_dist, scoring='neg_mean_squared_error',
                           cv=tr_val_partition, n_jobs=1, verbose=1)

# Training the model with the random-search
np.random.seed(123)
svrrs.fit(train_closest, train_minmax[:, 1200])

# Making predictions on the testing partition
y_test_pred = svrrs.predict(test_closest)

# And finally computing the test accuracy
print("Mean squared error of SVM with tuned hyperparameters:",
      metrics.mean_squared_error(y_test_pred, test_minmax[:, 1200]))
print("R-squared:", metrics.r2_score(y_test_pred, test_minmax[:, 1200]))

print("-----------------------------------------------------------------")
