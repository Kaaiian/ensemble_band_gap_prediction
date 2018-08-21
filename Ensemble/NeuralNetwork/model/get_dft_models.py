#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 20:11:38 2018

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
from ModelDFT import ModelDFT

# import code from the standard library 
import numpy as np
import pandas as pd
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# create objects from custom code
cv = CrossValidate()
display = DisplayData()
display.alpha = 0.1
display.markersize = 7
display.mfc='#073b4c'
modeldft = ModelDFT()

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

def combined_band_gap(database='combined'):
    # access the aflow data
    aflow_path = base_path + 'NeuralNetwork/Data/aflow-vectorized-train-data/'

    # access the materials project (mp) data
    mp_path = base_path + 'NeuralNetwork/Data/mp-vectorized-train-data/'

    # get the name of the property being used    
    prop = 'Band Gap'

    # acess the aflow data
    df_aflow_training_data = pd.read_csv( aflow_path + 'aflow vectorized '+prop+'.csv')
    df_aflow_training_data = df_aflow_training_data.iloc[:, 1:]

    # acess the materials project (mp) data
    df_mp_training_data = pd.read_csv(mp_path + 'mp vectorized '+prop+'.csv')
    df_mp_training_data = df_mp_training_data.iloc[:, 1:]

    if database == 'combined':
        # combined the aflow and dft data for learning
        df = pd.concat([df_aflow_training_data, df_mp_training_data], axis=0, ignore_index=True)

    elif database == 'aflow':
        df = df_aflow_training_data

    elif database == 'mp':
        df = df_mp_training_data

    return df, prop, database


# %%

def train_model(df, prop, database):
    modeldft = ModelDFT()
    if database == 'combined':
        epochs = 800
    elif database == 'aflow':
        epochs = 1500
    else:
        epochs = 1500
    
    modeldft.fit(df, prop, database, epochs=epochs, batch_size=epochs, evaluate=True)
    print(np.sqrt(modeldft.mse))
    print([modeldft.n1,
           modeldft.drop1,
           modeldft.n2,
           modeldft.drop2,
           modeldft.n3,
           modeldft.drop3,
           modeldft.lr,
           modeldft.decay])
    
    print('\a')
    
    y_exp_train_predicted = modeldft.predict(X_exp_train)
    y_exp_train_predicted.to_csv(base_path + 'NeuralNetwork/predictions/train/y_exp_train_predicted NN ' + database + ' ' + prop + '.csv', index=False)
    
    y_exp_test_predicted = modeldft.predict(X_exp_test)
    y_exp_test_predicted.to_csv(base_path + 'NeuralNetwork/predictions/test/y_exp_test_predicted NN ' + database + ' ' + prop + '.csv', index=False)
    
    modeldft.save_model()

# %%

for database in ['combined', 'aflow', 'mp']:
    df, prop, database = combined_band_gap(database)
    train_model(df, prop, database)
