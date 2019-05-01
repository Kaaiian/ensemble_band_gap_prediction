import sys
base_path = ''
sys.path.insert(0, base_path)

# read in custom functions and classes
from MachineLearningFunctions.MSE_ML_functions import CrossValidate
from MachineLearningFunctions.MSE_ML_functions import DisplayData

# import code from the standard library 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

# read in machine learning code
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import learning_curve, GridSearchCV, KFold, cross_val_predict

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

# define the SVR model parameters
svr = SVR(C=10, gamma=1)
gbr = GradientBoostingRegressor(n_estimators=500, max_depth=3)
rf = RandomForestRegressor(n_estimators=500, max_features='sqrt')

# scale the data
scaler = StandardScaler().fit(X_exp_train)
X_train = scaler.transform(X_exp_train)
normalizer = Normalizer().fit(X_train)
X_train = pd.DataFrame(normalizer.transform(X_train))
X_exp_test = scaler.transform(X_exp_test)
X_exp_test = pd.DataFrame(normalizer.transform(X_exp_test))

# %%

# read in experimental predictions
svr_train = pd.read_csv(base_path+'ExperimentalModels/SupportVectorRegression/predictions/svr_train.csv', header=None)
svr_test = pd.read_csv(base_path+'ExperimentalModels/SupportVectorRegression/predictions/svr_test.csv', header=None)

gbr_train = pd.read_csv(base_path+'ExperimentalModels/GradientBoostingRegression/predictions/gbr_train.csv', header=None)
gbr_test = pd.read_csv(base_path+'ExperimentalModels/GradientBoostingRegression/predictions/gbr_test.csv', header=None)

rf_train = pd.read_csv(base_path+'ExperimentalModels/RandomForestRegression/predictions/rf_train.csv', header=None)
rf_test = pd.read_csv(base_path+'ExperimentalModels/RandomForestRegression/predictions/rf_test.csv', header=None)

lr_train = pd.read_csv(base_path+'ExperimentalModels/LinearRegression/predictions/lr_train.csv', header=None)
lr_test = pd.read_csv(base_path+'ExperimentalModels/LinearRegression/predictions/lr_test.csv', header=None)

# read in DFT predictions'
aflow_train = pd.read_csv(base_path+'NeuralNetwork/predictions/train/y_exp_train_predicted NN aflow Band Gap.csv')
aflow_test = pd.read_csv(base_path+'NeuralNetwork/predictions/test/y_exp_test_predicted NN aflow Band Gap.csv')

mp_train = pd.read_csv(base_path+'NeuralNetwork/predictions/train/y_exp_train_predicted NN mp Band Gap.csv')
mp_test = pd.read_csv(base_path+'NeuralNetwork/predictions/test/y_exp_test_predicted NN mp Band Gap.csv')

combined_train = pd.read_csv(base_path+'NeuralNetwork/predictions/train/y_exp_train_predicted NN combined Band Gap.csv')
combined_test = pd.read_csv(base_path+'NeuralNetwork/predictions/test/y_exp_test_predicted NN combined Band Gap.csv')

# define the predicted DFT values
X_dft_train = pd.DataFrame(index=svr_train.index)
X_dft_train['combined'] = combined_train
X_train_w_dft = pd.concat([X_train, X_dft_train], axis=1)

X_dft_test = pd.DataFrame(index=svr_test.index)
X_dft_test['combined'] = combined_test
X_test_w_dft = pd.concat([X_exp_test, X_dft_test], axis=1)

# define the ensemble schemes
X_ensemble_train = pd.DataFrame(index=svr_train.index)
X_ensemble_train['svr'] = svr_train
#X_ensemble_train['gbr'] = gbr_train
#X_ensemble_train['rf'] = rf_train
X_ensemble_train['combined'] = combined_train

X_ensemble_test = pd.DataFrame(index=svr_test.index)
X_ensemble_test['svr'] = svr_test
#X_ensemble_test['gbr'] = gbr_test
#X_ensemble_test['rf'] = rf_test
X_ensemble_test['combined'] = combined_test

ensemble = SVR(C=150, gamma=0.003)

# %%
# =============================================================================
# make learning curve
# =============================================================================


def get_learning_curve_data(data_sizes, random_state=1):
    # save performace to lists
    test_mse_scheme2 = []
    test_mse_svr = []
    # increase the training data size with each loops
    for n_data in data_sizes:
        # get svr results
        X_train_sampled = X_train.sample(frac=n_data, random_state=random_state)
        y_exp_train_sampled = y_exp_train.loc[X_train_sampled.index]
        svr.fit(X_train_sampled, y_exp_train_sampled)
        # generate prediction on training set via cross-validation
        y_train_predicted = pd.Series(cross_val_predict(svr, X_train_sampled, y_exp_train_sampled, cv=10), index = X_train_sampled.index)
        # generate prediction on test set
        y_test_prediction = pd.Series(svr.predict(X_exp_test), index = X_exp_test.index)
        # save SVR performance
        test_mse_svr.append(mean_squared_error(y_exp_test, y_test_prediction))
        # ensemble svr prediction with DFT prediction
        X_ensemble_train_sampled = pd.concat([y_train_predicted, X_dft_train.loc[X_train_sampled.index]], axis=1)
        # define and fit ensemble model
        ensemble = SVR(C=150, gamma=0.003)
        ensemble.fit(X_ensemble_train_sampled, y_exp_train_sampled)
        # predict test set values
        X_ensemble_test = pd.concat([y_test_prediction, X_dft_test], axis=1)
        y_test_prediction_ensemble = pd.Series(ensemble.predict(X_ensemble_test))
        # save ensemble performance
        test_mse_scheme2.append(mean_squared_error(y_exp_test, y_test_prediction_ensemble))
    # print results with each loop
    print(np.array(test_mse_scheme2) - np.array(test_mse_svr))
    return test_mse_svr, test_mse_scheme2

# generate the data over different random shuffles
def get_lc_data_w_error_bounds(data_sizes):
    list_svr = []
    list_scheme3 = []
    for i in range(5):
        # get results for random state i
        test_mse_svr, test_mse_scheme3 = get_learning_curve_data(data_sizes, random_state=i)
        # save the svr and scheme3 results
        list_svr.append(test_mse_svr)
        list_scheme3.append(test_mse_scheme3)
    df_svr = pd.DataFrame(list_svr)
    df_scheme3 = pd.DataFrame(list_scheme3)
    return df_svr, df_scheme3

# define the size of training data for each loop
data_sizes = np.logspace(-2, 0, 20)
# return results for all random shuffles
df_svr, df_scheme3 = get_lc_data_w_error_bounds(data_sizes)

# %%

# plot the data
def plot_lc_data(df_svr, df_scheme3):
    # calculate the mean and std for SVR
    svr_scores_mean = np.sqrt(df_svr).mean()
    svr_scores_std = np.sqrt(df_svr).std()

    # calculate the mean and std for scheme3
    scheme3_scores_mean = np.sqrt(df_scheme3).mean()
    scheme3_scores_std = np.sqrt(df_scheme3).std()

    # convert data fraction to number of data in training set
    n_data = data_sizes * len(X_train)

    # define the figure
    fig, ax = plt.subplots(figsize=(7, 7))
    font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 18}
    plt.rc('font', **font)
    # plot the mean and std values (shows the background)
    plt.fill_between(n_data, svr_scores_mean - svr_scores_std,
                         svr_scores_mean + svr_scores_std, alpha=0.5, color='#2D2D9B')
    plt.fill_between(n_data, scheme3_scores_mean - scheme3_scores_std,
                         scheme3_scores_mean + scheme3_scores_std, alpha=0.25, color='#f44141')

    # plot the points at the mean values
    plt.plot(n_data, svr_scores_mean, linestyle='-', color='#2D2D9B', marker='o', mfc='white', label='SVR model')
    plt.plot(n_data, scheme3_scores_mean, linestyle='-', color='#f44141', marker='o', mfc='white', label='Ensemble of DFT\n& SVR models')

    plt.xlabel("Number of Examples")
    plt.ylabel("Root-Mean-Square Error")

    plt.legend(loc="best")
    plt.tick_params(direction='in', length=10, bottom=True, top=True, left=True, right=True)
    plt.tick_params(which='minor', direction='in', length=5, bottom=True, top=True, left=True, right=True)
    plt.ylim(0.5, 1.4)

    ax.loglog()
    ax.set_yticks([0.6, 0.8, 1, 1.2, 1.4])
    ax.set_yticks([], minor=True)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

    # save the figure here if you want.
    plt.savefig('figures/learning curve.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/learning curve.eps', bbox_inches='tight')
    plt.show()

# plot the data
plot_lc_data(df_svr, df_scheme3)
