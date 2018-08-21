#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 20:17:16 2018

@author: steven
"""
# =============================================================================
# # YOU NEED TO SET THE PATH TO MATCH THE LOCATION OF THE Ensemble FOLDER
# =============================================================================
import sys
## base_path = r'location of the folder Esemble'
base_path = r'/home/steven/Research/PhD/DFT Ensemble Models/publication code/Ensemble/'
sys.path.insert(0, base_path)

# read in custom functions and classes
from MachineLearningFunctions.MSE_ML_functions import CrossValidate
from MachineLearningFunctions.MSE_ML_functions import DisplayData


# import code from the standard library 
import numpy as np
import pandas as pd
import os
import time

# read in machine learning code
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.externals import joblib

# create objects from custom code
cv = CrossValidate()
display = DisplayData()
display.alpha = 1
display.markersize = 8
display.mfc='w'

# %%

# read in the training split for experimental data
df_exp_train = pd.read_csv(base_path+'ExperimentalData/df_exp_train.csv')
X_exp_train = df_exp_train.iloc[:,1:-1]
y_exp_train = df_exp_train.iloc[:,-1]

# read in the test split for experimental data
df_exp_test = pd.read_csv(base_path+'ExperimentalData/df_exp_test.csv')
X_exp_test = df_exp_test.iloc[:,1:-1]
y_exp_test = df_exp_test.iloc[:,-1]

# %%

svr = SVR(C=10, gamma=1) # r2, mse: 0.818253504446 0.381034083801
gbr = GradientBoostingRegressor(n_estimators=150, max_depth=7) # r2, mse: 0.792820362553 0.434355023441
rf = RandomForestRegressor(n_estimators=150, max_features='sqrt') # r2, mse: 0.78243767623 0.456122471329
lr = LinearRegression()

model = lr

y_actual, y_predicted, metrics, data_index = cv.cross_validate(X_exp_train, y_exp_train, model, N=5, random_state=1, scale_data=True)
display.actual_vs_predicted(y_actual, y_predicted)

r2, mse = r2_score(y_actual, y_predicted), mean_squared_error(y_actual, y_predicted)
print('r2, mse:', r2, mse)

# %%
models = [svr, gbr, rf, lr]
names = ['svr', 'gbr', 'rf', 'lr']
recorded_cv = []
def train_models():

    for model, name in zip(models, names):
    #    cross_validate(self, X, y, model, N=5, random_state=None, scale_data=False)
        y_actual, y_predicted, metrics, data_index = cv.cross_validate(X_exp_train, y_exp_train, model, N=5, random_state=1, scale_data=True)
        
        if name == 'svr':
            path = 'ExperimentalModels/SupportVectorRegression/'
        elif name == 'gbr':
            path = 'ExperimentalModels/GradientBoostingRegression/'
        elif name == 'rf':
            path = 'ExperimentalModels/RandomForestRegression/'
        elif name == 'lr':
            path = 'ExperimentalModels/LinearRegression/'
        else:
            print('error!')

        model.fit(X_exp_train, y_exp_train)
        y_test_prediction = pd.DataFrame(model.prediction(X_exp_test))
        display.actual_vs_predicted(y_actual, y_predicted, data_label= name + ' prediction', save=True, save_name=base_path + path + 'figures/' + name)
        y_predicted.to_csv(base_path + path + 'predictions/' + name + '_train.csv', index=False)
        y_test_prediction.to_csv(base_path + path + 'predictions/' + name + '_test.csv', index=False)
        joblib.dump(model, base_path + path + 'model/' + name + '.pkl') 
        recorded_cv.append(metrics) 
        
    writer = pd.ExcelWriter(base_path + 'ExperimentalModels/model_metrics.xlsx')
    for metric, name in zip(recorded_cv, names):
        metric.to_excel(writer, sheet_name=name)

train_models()
