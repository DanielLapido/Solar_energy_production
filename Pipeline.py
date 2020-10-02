# PIPELINE TO DEAL WITH MISSING VALUES
# AUTHOR: DANIEL LAPIDO MARTINEZ

# The datasets do not contain any missing values. However this is usually not the case, therefore,
# a few of them are created artificially.
# A 10% of missing values (np.nan) are put at random places in the selected columns.

# Once the missing values are set, two pipelines are created to deal with them:
# the first one uses PCA, the second one uses feature selection (with SelectKBest)

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline


### Set seed for reproducibility
seed = 50559292
np.random.seed(seed)

### Split Between Training and Testing
train = pd.read_pickle('trainst1ns16.pkl')
test = pd.read_pickle('testst1ns16.pkl')
X_train = train.iloc[:, 0:300] #From variable 0 to 299 (First 4 forecasts variables)
Y_train = train.iloc[:, 1200]
X_test = test.iloc[:, 0:300]
Y_test = test.iloc[:, 1200]

print(X_train.tail()) #Check some results



###Creating NaNs
X_train.iloc[np.random.choice(len(X_train.index),int((len(X_train.index))*0.1),replace=False) ,np.random.choice(len(X_train.columns),int((len(X_train.columns)+1)*0.1), replace=False)]=np.nan
print(pd.isnull(X_train).sum().sum()) #Check the amount of NaNs is 10% of 300 times 10% of 4380 equal 13140

np.random.seed(seed)
X_test.iloc[np.random.choice(len(X_test.index),int((len(X_test.index))*0.1),replace=False) ,np.random.choice(len(X_test.columns),int((len(X_test.columns)+1)*0.1), replace=False)]=np.nan
print(pd.isnull(X_test).sum().sum())

####Steps for the Pipelines

### PreProcessing Steps
MedImputation =  SimpleImputer(strategy='median')
Remove = VarianceThreshold() #Remove constant variables
Scaler = MinMaxScaler()

### Selection Steps
Selection = SelectKBest(f_regression)
pca = PCA()

### Regressor Step
KNN = KNeighborsRegressor()

### First Pipeline
PipelineSelection = Pipeline([("Imputation", MedImputation), ("Removal", Remove), ("Scaling", Scaler), ("Selection", Selection), ("KNN", KNN)])

### Second Pipeline
PipelinePCA = Pipeline ([("Imputation", MedImputation), ("Removal", Remove), ("Scaling", Scaler), ("Selection", pca), ("KNN", KNN)])

#Check Regressor Pipelines Definitions
print(PipelineSelection)
print(PipelinePCA)


### Optimize Parameters in Both Pipelines

#Predefined Validation Set for Hyper-Paramter Tuning
ValidationIndices = np.zeros(X_train.shape[0])
ValidationIndices[:10*365] = -1 # 10 years of training and 2 of validation set
ValidationPartition = PredefinedSplit(ValidationIndices)


### Variable Selection Hyper-Parameter Tuning
gridSelection = {'Selection__k': (list(range(1,41,1)))} #Grid for number of variables between 1 and 40
gridPCA = {'Selection__n_components': (list(range(1,41,1)))}

PipelineSelection_Grid = GridSearchCV(PipelineSelection, gridSelection, scoring = 'neg_mean_squared_error', cv=ValidationPartition , n_jobs=1, verbose=1)
PipelinePCA_Grid = GridSearchCV(PipelinePCA, gridPCA, scoring = 'neg_mean_squared_error', cv=ValidationPartition , n_jobs=1, verbose=1)

PipelineSelection_Grid.fit(X_train, Y_train) #Optimize hyper-parameters
PipelinePCA_Grid.fit(X_train, Y_train)

### Setting Hypperparameters
KSelection = PipelineSelection_Grid.best_params_
KPCA = PipelinePCA_Grid.best_params_
PipelineSelection = PipelineSelection.set_params(**(KSelection))#Fix the number of variables selected to 6
PipelinePCA = PipelinePCA.set_params(**(KPCA)) #Fix the number of principal components to 7

print(PipelineSelection_Grid.best_params_)
print(PipelinePCA_Grid.best_params_)


### Number of Neighbors Hyper-Parameter Tuning
gridNeighbors = {'KNN__n_neighbors': list(range(1,21,1))}  # Explore the number of neighbors from 1 up to 20

PipelineSelectionNeighbors_Grid = GridSearchCV(PipelineSelection, gridNeighbors, scoring = 'neg_mean_squared_error', cv=ValidationPartition , n_jobs=1, verbose=1)
PipelinePCANeighbors_Grid = GridSearchCV(PipelinePCA, gridNeighbors, scoring = 'neg_mean_squared_error', cv=ValidationPartition , n_jobs=1, verbose=1)

PipelineSelectionNeighbors_Grid.fit(X_train, Y_train)  # Optimize hyper-parameter in both cases
PipelinePCANeighbors_Grid.fit(X_train, Y_train)

#Check
print(PipelineSelectionNeighbors_Grid.best_params_)  # 7 neighbors are enough in the case of variable Selection

print(PipelinePCANeighbors_Grid.best_params_)  # 18 neighbors are selected in the case of PCA,
                                               # probably the grid is too restrictive in this case.



##Model evaluation
Y_SelectionTest_Pred = PipelineSelectionNeighbors_Grid.predict(X_test)
Y_PCATest_Pred = PipelinePCANeighbors_Grid.predict(X_test)

print("Mean squared error of feature selection pipeline:", metrics.mean_squared_error(Y_SelectionTest_Pred, Y_test))
print("Mean squared error of PCA pipeline:", metrics.mean_squared_error(Y_PCATest_Pred, Y_test))
# We see that the models whose attributes are selected based on principal components produces a slighty better fit
# of the testing set. This makes sense, since it is able to capture more informationn with less overfitting than
# simple variable selection when there are variables not very informative.

# We can check the R2 just in order to obtain a more comparable metric
print("R-squared of feature selection pipeline:", metrics.r2_score(Y_SelectionTest_Pred, Y_test))
print("R-squared of PCA pipeline:", metrics.r2_score(Y_PCATest_Pred, Y_test))
# When compared with the results of the previous assignment we see that the fit improves disregarding whether PCA
# or variable selection is employed, despite the fact that the desing matrix contained missing values.
# Again this is a sensible result since we are taking into account more information (300 variables vs 75) than before
# and the missing values only represented a 1% of the data set.


print(X_test.iloc[:,PipelineSelectionNeighbors_Grid.best_estimator_.named_steps['Selection'].get_support('TRUE')].head())
# Finally, it is interesting to note that the only variable selected by means of f_regression is that of the average
# short-wave radiation. This variable prediction for the end of the day is selected for the 4 predictions
# and the previous prediction are alse taken into account for the closest case.