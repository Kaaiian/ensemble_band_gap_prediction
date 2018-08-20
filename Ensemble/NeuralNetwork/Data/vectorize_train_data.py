import sys
# allow access to custom machine learning functions
sys.path.insert(0, r'F:\Sparks Group\Machine Learning Functions')

# custom imports
from MSE_ML_functions import VectorizeFormulaMinimal

# standard imports
import numpy as np
import pandas as pd
import os

vfm = VectorizeFormulaMinimal()

def save_for_training(df, prop, db):
    X, y, formula = vfm.generate_features(df)
    df_conc = pd.concat([formula, X, y], axis=1)
    df_conc.to_csv(db+'-vectorized-train-data/'+db+' vectorized '+prop+'.csv', index=False)
    return df_conc

def get_vectorized_csv(db):
    for file in os.listdir(db+'-train-data'):
        prop = file[9:-4]
        df = pd.read_csv(db+r'-train-data/'+file, keep_default_na=False)
        save_for_training(df, prop, db)

get_vectorized_csv('mp')
get_vectorized_csv('aflow')
