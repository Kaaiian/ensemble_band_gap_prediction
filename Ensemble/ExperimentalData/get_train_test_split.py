#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:16:58 2018

@author: steven
"""
# =============================================================================
# # YOU NEED TO SET THE PATH TO MATCH THE LOCATION OF THE Ensemble FOLDER
# =============================================================================
import sys
## base_path = r'location of the folder Esemble'
base_path = r'/home/steven/Research/PhD/DFT Ensemble Models/publication code/Ensemble/'
sys.path.insert(0, base_path)

from MachineLearningFunctions.MSE_ML_functions import VectorizeFormulaMinimal
import pandas as pd

from sklearn.model_selection import train_test_split

vf = VectorizeFormulaMinimal()

# read in band gap data from Zhou et al. publication:
# J. Phys. Chem. Lett., 2018, 9 (7), pp 1668–1673
#DOI: 10.1021/acs.jpclett.8b00124
#Publication Date (Web): March 13, 2018
df_band_gap = pd.read_excel('band_gap_download.xlsx') # excel sheet was edited to fix formula "GaAs0.1P0.9G1128" to "GaAs0.1P0.9"

# take the average of duplicte composition entries
df_band_gap = df_band_gap.groupby('composition').mean().reset_index()

# separate the metal and non-metal compounds
df_band_gap_non_metal = df_band_gap[df_band_gap['Eg (eV)'] > 0]
df_band_gap_metal = df_band_gap[df_band_gap['Eg (eV)'] == 0]

# randomly select train and test split
df_train = df_band_gap_non_metal.sample(frac=0.8, random_state=256)
df_test = df_band_gap_non_metal.iloc[~df_band_gap_non_metal.index.isin(df_train.index.values)]

# rename columns for use with feature generation
df_train.columns = ['formula', 'target']
df_test.columns = ['formula', 'target']

# save the formula and band gap for train and test splits
df_train.to_csv('train.csv')
df_test.to_csv('test.csv')

# generate features for both train and test split
X_train, y_train, formula_train = vf.generate_features(df_train)
X_test, y_test, formula_test = vf.generate_features(df_test)

# save the vectorized train and test data
df_exp_train = pd.concat([formula_train, X_train, y_train], axis=1)
df_exp_test = pd.concat([formula_test, X_test, y_test], axis=1)

df_exp_train.to_csv('df_exp_train.csv', index=False)
df_exp_test.to_csv('df_exp_test.csv', index=False)
