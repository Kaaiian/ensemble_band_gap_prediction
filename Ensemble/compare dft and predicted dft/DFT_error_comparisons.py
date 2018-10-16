#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 08:21:11 2018

@author: steven
"""

import pymatgen as mg
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import r2_score, mean_squared_error

df_aflow = pd.read_csv('aflow Band Gap.csv')
df_aflow_train = pd.read_csv('aflow train Band Gap.csv')

df_mp = pd.read_csv('mp Band Gap.csv')
df_mp_train = pd.read_csv('mp train Band Gap.csv')

df_exp = pd.read_csv('band_gap_download.csv')
df_exp.drop_duplicates(['formula'], inplace=True)
df_exp.index = df_exp['formula']

df_exp_aflow = df_aflow[~df_aflow['formula'].isin(df_aflow_train['formula'])]
df_exp_aflow.index = df_exp_aflow['formula']

df_exp_mp = df_mp[~df_mp['formula'].isin(df_mp_train['formula'])]
df_exp_mp.index = df_exp_mp['formula']

df_exp_test = pd.DataFrame()
df_exp_test['aflow'] = df_exp_aflow['target']
df_exp_test['mp'] = df_exp_mp['target']

df_combined_cv = pd.read_csv('NN_combined_act_vs_pred.csv')

#df_exp['mp'] = df_exp_mp['target']

# %%
def match_df_compositions(df1, df2):
    '''
    Compares the fractional composition between the formula in two dataframes.

    Parameters
    ----------
    df1: pd.DataFrame()
        DataFrame with a 'formula' column
    df2: pd.DataFrame()
        vector containing column names 'formula'

    Return
    ----------
    df1, df2: original dataframes limited to rows that had matched formula
    '''
    # copy the data frames so we don't overide old dataframe
    df1 = df1.copy()
    df1.reset_index(drop=True, inplace=True)

    df2 = df2.copy()
    df2.reset_index(drop=True, inplace=True)

    # find the franctional composition for all formula and store in df
    for j in range(len(df1)):
        formula1 = df1['formula'].loc[j]
        c1 = mg.Composition(formula1).fractional_composition.as_dict()
        df1.loc[j, 'comp'] = str(c1)
    try:
        for k in range(len(df2)):
            formula2 = df2['formula'].loc[k]
            c2 = mg.Composition(formula2).fractional_composition.as_dict()
            df2.loc[k, 'comp'] = str(c2)
    except:
        print('COMPOSITION ERROR: ', formula2, k)

    # remove duplicates in the fractional composition
    df1.drop_duplicates(subset=['comp'], keep=False, inplace=True)
    df2.drop_duplicates(subset=['comp'], keep=False, inplace=True)

    # compare dataframes and return the matching indexis
    df1 = df1[df1['comp'].isin(df2['comp'].values)]
    df2 = df2[df2['comp'].isin(df1['comp'].values)]

    # make sure both dataframes match up by sorting on the matched value
    df1.sort_values(by=['comp'], inplace=True)
    df2.sort_values(by=['comp'], inplace=True)

    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    # make a new dataframe that holds the formula and experimental values
    df_1 = pd.DataFrame()
    df_1['formula'] = df1['formula'].loc[df1['comp'] == df2['comp']]
    df_1 = df1.loc[df2['comp'] == df1['comp']]

    # make a dataframe that holds the DFT values on the same index as df above
    df_2 = pd.DataFrame()
    df_2['formula'] = df2['formula'].loc[df1['comp'] == df2['comp']]
    df_2 = df2.loc[df1['comp'] == df2['comp']]

    return df_1, df_2

df_matched_aflow_exp, df_matched_aflow = match_df_compositions(df_exp, df_exp_aflow)
df_matched_mp_exp, df_matched_mp = match_df_compositions(df_exp, df_exp_mp)

# %%

df_matched = pd.DataFrame()


df_matched_exp = pd.concat([df_matched_aflow_exp, df_matched_mp_exp])
index = df_matched_exp['formula']
df_matched_exp.index = index
df_matched_exp = df_matched_exp[~df_matched_exp.index.duplicated(keep='first')]
df_matched_exp.drop(['formula', 'comp'], axis=1, inplace=True)
df_matched_exp.columns = ['exp']


df_matched['exp'] = df_matched_exp['exp']
# %%

# # score- dft vs exp
#score_aflow = r2_score(df_matched_aflow_exp['target'], df_matched_aflow['target'])
#rmse_aflow = np.sqrt(mean_squared_error(df_matched_aflow_exp['target'], df_matched_aflow['target']))
#
#print('aflow -- r2, rmse:', score_aflow, rmse_aflow)
#
#score_mp = r2_score(df_matched_mp_exp['target'], df_matched_mp['target'])
#rmse_mp = np.sqrt(mean_squared_error(df_matched_mp_exp['target'], df_matched_mp['target']))
#
#print('mp -- r2, rmse:', score_mp, rmse_mp)

df_matched_dft = pd.concat([df_matched_aflow, df_matched_mp])
df_matched_dft.index = index
df_matched_dft = df_matched_dft[~df_matched_dft.index.duplicated(keep='first')]
df_matched_dft.drop(['formula', 'comp'], axis=1, inplace=True)
df_matched_dft.columns = ['dft']


df_matched['dft'] = df_matched_dft['dft']

# %%
df_exp_train = pd.read_csv('df_exp_train.csv')
df_exp_test = pd.read_csv('df_exp_test.csv')

df_pred_train = pd.read_csv('y_exp_train_predicted NN combined Band Gap.csv')
df_pred_train.index = df_exp_train['formula']

df_pred_test = pd.read_csv('y_exp_test_predicted NN combined Band Gap.csv')
df_pred_test.index = df_exp_test['formula']

df_matched_pred = pd.concat([df_pred_train, df_pred_test])
df_matched_pred.columns = ['predicted']

df_matched['predicted'] = df_matched_pred['predicted']
# %%

score_mp = r2_score(df_matched['exp'], df_matched['dft'])
rmse_mp = np.sqrt(mean_squared_error(df_matched['exp'], df_matched['dft']))
print('(dft) r2, rmse:', score_mp, rmse_mp)


score_mp = r2_score(df_matched['exp'], df_matched['predicted'])
rmse_mp = np.sqrt(mean_squared_error(df_matched['exp'], df_matched['predicted']))
print('(predicted) r2, rmse:', score_mp, rmse_mp)

plt.figure(1, figsize=(7, 7))
font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 18}
plt.rc('font', **font)
plt.tick_params(direction='in', length=10, bottom=True, top=True, left=True, right=True)
plt.plot(df_matched['exp'], df_matched['dft'],  color='#CC99CC', marker='D', linestyle='None', mew=2, markerfacecolor=None, alpha=0.6,  markersize=10, markeredgewidth=1)
plt.plot(df_matched['exp'], df_matched['predicted'],  color='#708238', marker='o', linestyle='None', mew=2, markerfacecolor=None, alpha=0.5,  markersize=10, markeredgewidth=1)
plt.plot(df_matched['exp'], df_matched['exp'], 'k--')
plt.legend(['DFT (aflow and MP)', 'ML predicted of DFT', 'Ideal Performance'])
plt.xlabel('Experimental Band Gap (eV)', fontsize=22)
plt.ylabel('Predicted Band Gap (eV)', fontsize=22)
plt.xlim((0, 12))
plt.ylim((0,12))
plt.show()


# %%

plt.figure(2, figsize=(7, 7))
font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 18}
plt.rc('font', **font)
plt.tick_params(direction='in', length=10, bottom=True, top=True, left=True, right=True)
plt.plot(df_matched['dft'], df_matched['predicted'],  color='#708238', marker='o', linestyle='None', mew=2, markerfacecolor=None, alpha=0.5,  markersize=10, markeredgewidth=1)
plt.plot(df_matched['exp'], df_matched['exp'], 'k--')
plt.legend(['ML predicted of DFT', 'Ideal Performance'])
plt.xlabel('DFT Band Gap (eV)', fontsize=22)
plt.ylabel('Predicted Band Gap (eV)', fontsize=22)
plt.xlim((0, 10))
plt.ylim((0,10))
plt.show()

# %%
residual_DFT = df_matched['exp'] - df_matched['dft']
residual_pred = df_matched['exp'] - df_matched['predicted']



fig = plt.figure(1, figsize=(3, 3))
plot1 = sns.distplot(residual_DFT, color='r', label='Calculated DFT')
plt.tick_params(direction='in', length=10, bottom=True, top=True, left=True, right=True)
kde = plot1.get_lines()
plt.setp(kde, linewidth=4)
plt.legend(['DFT residual'])

fig = plt.figure(2, figsize=(3, 3))
font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 18}
plt.rc('font', **font)
plot1 = sns.distplot(residual_pred, color='b')
line_styles = ['--']
colors = [ 'b']
i = 0
for line in plot1.get_lines():
    line.set_linestyle(line_styles[i])
    line.set_linewidth(4)
    line.set_color(colors[i])
    i += 1
plt.legend(['ML residual'])
plt.show()

fig = plt.figure(3, figsize=(3, 3))
font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 18}
plt.rc('font', **font)
plot1 = sns.kdeplot(residual_pred, color='b')
plot2 = sns.kdeplot(residual_DFT, color='r')
line_styles = ['-', '--']
colors = ['r', 'b']
i = 0
for line in plot1.get_lines():
    line.set_linestyle(line_styles[i])
    line.set_linewidth(4)
    line.set_color(colors[i])
    i += 1
plt.legend(['DFT residual', 'predicted DFT residual'])
plt.show()

# %%

fig = plt.figure(1, figsize=(8, 8))

ax1 = fig.add_subplot(221)
ax1 = sns.distplot(residual_DFT, color='r', label='Calculated DFT')
plt.tick_params(direction='in', length=10, bottom=True, top=True, left=True, right=True)
kde = plot1.get_lines()
plt.setp(kde, linewidth=4)

ax2 = fig.add_subplot(222)
ax2 = sns.distplot(residual_pred, color='b', label='Calculated DFT')
plt.tick_params(direction='in', length=10, bottom=True, top=True, left=True, right=True)
kde = plot1.get_lines()
plt.setp(kde, linewidth=4)


plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)

ax1 = fig.add_subplot(223)
ax1 = sns.distplot(residual_DFT, color='r', label='Calculated DFT')
plt.tick_params(direction='in', length=10, bottom=True, top=True, left=True, right=True)
kde = plot1.get_lines()
plt.setp(kde, linewidth=4)

ax2 = fig.add_subplot(224)
ax2 = sns.distplot(residual_pred, color='b', label='Calculated DFT')
plt.tick_params(direction='in', length=10, bottom=True, top=True, left=True, right=True)
kde = plot1.get_lines()
plt.setp(kde, linewidth=4)

#ax3 = fig.add_subplot(212)
#ax3 = sns.kdeplot(residual_pred, color='b')
#plot2 = sns.kdeplot(residual_DFT, color='r')
#plt.tick_params(direction='in', length=10, bottom=True, top=True, left=True, right=True)
#kde = plot1.get_lines()
#plt.setp(kde, linewidth=4)


# %%

df_cv = pd.read_csv('NN_combined_act_vs_pred.csv')
plt.figure(2, figsize=(6, 6))
font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 18}
plt.rc('font', **font)
plt.tick_params(direction='in', length=10, bottom=True, top=True, left=True, right=True)
plt.plot(df_cv['actual'], df_cv['predicted'],  color='#CC99CC', marker='o', linestyle='None', mew=2, markerfacecolor=None, alpha=0.05,  markersize=10, markeredgewidth=1)
plt.plot(df_matched['dft'], df_matched['predicted'],  color='#708238', marker='X', linestyle='None', mew=2, markerfacecolor=None, alpha=0.5,  markersize=10, markeredgewidth=1)
plt.plot([0, 10], [0, 10], 'k--')
plt.legend(['CV predicted values', 'ML predicted of DFT', 'Ideal Performance'])
plt.xlabel('DFT Band Gap (eV)', fontsize=22)
plt.ylabel('Predicted Band Gap (eV)', fontsize=22)
plt.xlim((0, 10))
plt.ylim((0,10))
plt.show()

plt.plot(df_combined_cv['actual'], df_combined_cv['predicted'], 'rx', alpha=0.1)
