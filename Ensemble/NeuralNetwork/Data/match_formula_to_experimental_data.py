import sys
# allow access to custom machine learning functions
sys.path.insert(0, r'F:\Sparks Group\Machine Learning Functions')

# custom imports
from MSE_ML_functions import VerifyCompositions

import pymatgen as mg
import pandas as pd
import matplotlib.pyplot as plt
import os

vc = VerifyCompositions()

# %%

df_exp_eg = pd.read_csv('Experimental_Band_Gap.csv')

# %%

for file in os.listdir('aflow-data'):
    prop = file[6:-4]
    df_aflow = pd.read_csv(r'aflow-data/'+file)

    df1_aflow, df2_aflow = vc.match_df_compositions(df_exp_eg, df_aflow)

    df_aflow_train = df_aflow[~df_aflow['formula'].isin(list(df2_aflow['formula'].values))]
    df_aflow_train.to_csv('aflow-train-data/aflow train '+prop+'.csv', index=False)


# %%

for file in os.listdir('mp-data'):
    prop = file[3:-4]
    df_mp = pd.read_csv(r'mp-data/'+file, keep_default_na=False)

    df1_mp, df2_mp = vc.match_df_compositions(df_exp_eg, df_mp)

    df_mp_train = df_mp[~df_mp['formula'].isin(list(df2_mp['formula'].values))]
    df_mp_train.to_csv('mp-train-data/mp train '+prop+'.csv', index=False)
