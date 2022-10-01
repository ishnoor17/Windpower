import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use("MacOSX")
#matplotlib.use("TkAgg")
#matplotlib.use('QtAgg')
#https://www.kaggle.com/datasets/theforcecoder/wind-power-forecasting
import matplotlib.pyplot as plt
#from dtreeviz.trees import * -- GET PACKAGES -- to see the splitting, interpretable
#from treeinterpreter import treeinterpreter
#from waterfall_chart import plot as waterfall

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
#plt.plot([1, 2, 3, 4])
#plt.ylabel('some numbers')
#plt.show()
# Baseline/ First Model
#iris = load_iris()

#X = iris.data  # x is a two dimentional array -- from the wind data
#y = iris.target # y is a column of values -- target column

file_name = r"/Volumes/GoogleDrive/My Drive/Summer Projects 10th grade/UMD project windpower/Turbine_Data.csv.zip"
wind_data = pd.read_csv(file_name,low_memory=False)
print(wind_data.head(5))
target, features = wind_data.columns[1], wind_data.columns[2:] # which columns will be used for predictions, indepent varaibles that are given
X = wind_data[features]
y = wind_data[target]

I = y.notnull()
X, y = X[I], y[I] # takes care of the missing values

# Split dataset into train and test -- also has random number generator
# Did not initialize that will give different data everytime you run it
# determinants
# Sort the data in increasing order by time stamp
columns1, columns2 = X.select_dtypes(include=['number', ]).columns.tolist(), \
                     X.select_dtypes(exclude=['number', ]).columns.tolist()

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

my_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
my_imputer.fit(X_train[columns1]) # training imputer
X_train[columns1] = my_imputer.transform(X_train[columns1]) # imputing the training data
X_valid[columns1] = my_imputer.transform(X_valid[columns1]) # imputing the validation data
X_test[columns1] = my_imputer.transform(X_test[columns1]) # imputing the test data

# just transport not split
#X_train_imputed = my_imputer.fit_transform(X_train.select_dtypes(include=['number', ]))
#X_train_unchanged = X_train.select_dtypes(exclude=['number', ])
#imp_median = SimpleImputer(strategy='most_frequent')
#imp_median.fit(wind_data)
#imputed_train_df = imp_median.transform(wind_data)

#wind_data.isnull().sum()

#columns = ['AmbientTemperatue' ,'BearingShaftTemperature', 'Blade1PitchAngle', 'Blade2PitchAngle' ,'Blade3PitchAngle','GearboxBearingTemperature', 'GearboxOilTemperature','GeneratorRPM','GeneratorWinding1Temperature','GeneratorWinding2Temperature','HubTemperature','MainBoxTemperature','NacellePosition','ReactivePower','RotorRPM','TurbineStatus', 'WindDirection','WindSpeed']

#for n in columns:
    #wind_data[n].fillna(wind_data[n].median(),inplace = True)

from sklearn.tree import DecisionTreeRegressor

est = DecisionTreeRegressor(max_depth=16)
#neigh = KNeighborsRegressor(n_neighbors=5)
#forest = RandomForestRegressor(max_depth=2, verbose=2)

# will memorize, but not evaluate

est.fit(X_train[columns1], y_train)
#neigh.fit(X_train[columns1], y_train)
#forest.fit(X_train[columns1], y_train)

# fit the regressor to the training data
y_pred = est.predict(X_valid[columns1]) # make predictions on the validation datat
#y_pred2 = neigh.predict(X_valid[columns1]) # make predictions on the validation datat
v = y_valid.to_numpy().ravel()
#plt.plot(v)
#plt.show()

mse = mean_squared_error(y_valid, y_pred) # between predicted output and the actual true validation data y_valid
mae = mean_absolute_error(y_valid, y_pred)
mdae = median_absolute_error(y_valid, y_pred)

print(f'mse on validation data = {mse:.4f}')
# plot of the errors
# x axis -- y_valid
# lowest to highest power
#plt.plot(x=y_valid, y=y_pred-y_valid)
S = np.argsort(y_valid)
print(S)
e = y_pred - y_valid
print(e)
plt.plot(e, ls='-')
#plt.plot(y_valid[S], e[S])
plt.hist(e) # to see if this bell curve has the normal distribution

# pandas dataframe
# one of the columns will be a target data that we want to predict
# other columns is what we want to predict

#regdict = {
 # "DecisionTreeRegressor": ,
 # "RandomForestRegressor": ,
 # "KNeighborsRegressor":
#}

# for loop, for every key value pair in the dictionary, fit the reg and pred valid, fit regressors on the train data and pred on valid