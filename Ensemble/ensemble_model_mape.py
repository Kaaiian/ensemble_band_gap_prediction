import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from MachineLearningFunctions.MSE_ML_functions import CrossValidate
from MachineLearningFunctions.MSE_ML_functions import DisplayData

# create objects from custom code
cv = CrossValidate()


def calc_mape(y_act, y_pred):
    ape = []
    for act, pred in zip(y_act, y_pred):
        if act != 0:
            val = abs(act-pred)/act+0.001 * 100
            ape.append(val)
        else:
            print('error')
    mape = sum(ape)/len(ape)
#    plt.plot(y_act, ape, 'ro')
#    plt.show()
    return mape

def return_metrics(y_true, y_pred, X_ensemble):

    scores = [
            r2_score(y_true, X_ensemble['svr']),
#            r2_score(y_true, X_ensemble['gbr']),
#            r2_score(y_true, X_ensemble['rf']),
            r2_score(y_true, y_pred)]

    rmse = [
            np.sqrt(mean_squared_error(y_true, X_ensemble['svr'])),
#            np.sqrt(mean_squared_error(y_true, X_ensemble['gbr'])),
#            np.sqrt(mean_squared_error(y_true, X_ensemble['rf'])),
            np.sqrt(mean_squared_error(y_true, y_pred))]

    mape = [
            calc_mape(y_true, X_ensemble['svr']),
#            calc_mape(y_true, X_ensemble['gbr']),
#            calc_mape(y_true, X_ensemble['rf']),
            calc_mape(y_true, y_pred)]
#    print(scores)
#    print(rmse)
#    print(mape)
    return scores, rmse, mape


# read in train and test data
df_exp_train = pd.read_csv('ExperimentalData/df_exp_train.csv')
df_exp_test = pd.read_csv('ExperimentalData/df_exp_test.csv')

y_exp_train = df_exp_train.iloc[:, -1]
y_exp_test = df_exp_test.iloc[:, -1]


# read in experimental predictions
svr_train = pd.read_csv('ExperimentalModels/SupportVectorRegression/predictions/svr_train.csv', header=None)
svr_test = pd.read_csv('ExperimentalModels/SupportVectorRegression/predictions/svr_test.csv', header=None)

gbr_train = pd.read_csv('ExperimentalModels/GradientBoostingRegression/predictions/gbr_train.csv', header=None)
gbr_test = pd.read_csv('ExperimentalModels/GradientBoostingRegression/predictions/gbr_test.csv', header=None)

rf_train = pd.read_csv('ExperimentalModels/RandomForestRegression/predictions/rf_train.csv', header=None)
rf_test = pd.read_csv('ExperimentalModels/RandomForestRegression/predictions/rf_test.csv', header=None)

lr_train = pd.read_csv('ExperimentalModels/LinearRegression/predictions/lr_train.csv', header=None)
lr_test = pd.read_csv('ExperimentalModels/LinearRegression/predictions/lr_test.csv', header=None)

# read in DFT predictions'
aflow_train = pd.read_csv('NeuralNetwork/predictions/train/y_exp_train_predicted NN aflow Band Gap.csv')
aflow_test = pd.read_csv('NeuralNetwork/predictions/test/y_exp_test_predicted NN aflow Band Gap.csv')

mp_train = pd.read_csv('NeuralNetwork/predictions/train/y_exp_train_predicted NN mp Band Gap.csv')
mp_test = pd.read_csv('NeuralNetwork/predictions/test/y_exp_test_predicted NN mp Band Gap.csv')

combined_train = pd.read_csv('NeuralNetwork/predictions/train/y_exp_train_predicted NN combined Band Gap.csv')
combined_test = pd.read_csv('NeuralNetwork/predictions/test/y_exp_test_predicted NN combined Band Gap.csv')


def scheme1():

    # create ensemble feature vector from model predictions
    X_ensemble_train = pd.DataFrame(index=svr_train.index)
    X_ensemble_train['svr'] = svr_train
    X_ensemble_train['gbr'] = gbr_train
    X_ensemble_train['rf'] = rf_train
    #X_ensemble_train['lr'] = lr_train
    #X_ensemble_train['aflow'] = aflow_train
    #X_ensemble_train['mp'] = mp_train
    #X_ensemble_train['combined'] = combined_train

    custom_axis = {}
    custom_axis['xlim_min'] = -0.3
    custom_axis['xlim_max'] = 11
    custom_axis['xlim_min'] = -0.3
    custom_axis['ylim_max'] = 11
    custom_axis['ticks'] = np.arange(-0, 6)*2

    display = DisplayData()
    display.alpha = 0.2
    display.markersize = 8
    display.mfc='w'
    display.edgewidth = 0

    svr = SVR(C=150, gamma=0.003)
    gmr = GradientBoostingRegressor(n_estimators=500, max_depth=3)
    rf = RandomForestRegressor(n_estimators=150)
    lr = LinearRegression()

    model = svr

    y_actual, y_predicted, metrics, data_index = cv.cross_validate(X_ensemble_train, y_exp_train, model, N=10, random_state=7, scale_data=False)

#    print('cross validation metrics:', metrics.T.mean())

    # # fit model if suitable results are found
    model.fit(X_ensemble_train, y_exp_train)

    # create ensemble feature vector from model predictions
    X_ensemble_test = pd.DataFrame(index=svr_test.index)
    X_ensemble_test['svr'] = svr_test
    X_ensemble_test['gbr'] = gbr_test
    X_ensemble_test['rf'] = rf_test
#    X_ensemble_test['lr'] = lr_test
    #X_ensemble_test['aflow'] = aflow_test
    #X_ensemble_test['mp'] = mp_test
#    X_ensemble_test['combined'] = combined_test

    y_ensemble = model.predict(X_ensemble_test)

    test_r2, test_rmse, test_mape = return_metrics(y_exp_test, y_ensemble, X_ensemble_test)
    return test_r2, test_rmse, test_mape


def scheme2():

    # create ensemble feature vector from model predictions
    X_ensemble_train = pd.DataFrame(index=svr_train.index)
    X_ensemble_train['svr'] = svr_train
    X_ensemble_train['gbr'] = gbr_train
    X_ensemble_train['rf'] = rf_train
    #X_ensemble_train['lr'] = lr_train
    #X_ensemble_train['aflow'] = aflow_train
    #X_ensemble_train['mp'] = mp_train
    X_ensemble_train['combined'] = combined_train

    custom_axis = {}
    custom_axis['xlim_min'] = -0.3
    custom_axis['xlim_max'] = 11
    custom_axis['xlim_min'] = -0.3
    custom_axis['ylim_max'] = 11
    custom_axis['ticks'] = np.arange(-0, 6)*2

    display = DisplayData()
    display.alpha = 0.2
    display.markersize = 8
    display.mfc='w'
    display.edgewidth = 0

    svr = SVR(C=150, gamma=0.003)
    gmr = GradientBoostingRegressor(n_estimators=500, max_depth=3)
    rf = RandomForestRegressor(n_estimators=150)
    lr = LinearRegression()

    model = svr

    y_actual, y_predicted, metrics, data_index = cv.cross_validate(X_ensemble_train, y_exp_train, model, N=10, random_state=7, scale_data=False)

#    print('cross validation metrics:', metrics.T.mean())

    # # fit model if suitable results are found
    model.fit(X_ensemble_train, y_exp_train)

    # create ensemble feature vector from model predictions
    X_ensemble_test = pd.DataFrame(index=svr_test.index)
    X_ensemble_test['svr'] = svr_test
    X_ensemble_test['gbr'] = gbr_test
    X_ensemble_test['rf'] = rf_test
    #X_ensemble_test['lr'] = lr_test
    #X_ensemble_test['aflow'] = aflow_test
    #X_ensemble_test['mp'] = mp_test
    X_ensemble_test['combined'] = combined_test

    y_ensemble = model.predict(X_ensemble_test)

    test_r2, test_rmse, test_mape = return_metrics(y_exp_test, y_ensemble, X_ensemble_test)
    return test_r2, test_rmse, test_mape


def scheme3():

    # create ensemble feature vector from model predictions
    X_ensemble_train = pd.DataFrame(index=svr_train.index)
    X_ensemble_train['svr'] = svr_train
#    X_ensemble_train['gbr'] = gbr_train
#    X_ensemble_train['rf'] = rf_train
    #X_ensemble_train['lr'] = lr_train
    #X_ensemble_train['aflow'] = aflow_train
    #X_ensemble_train['mp'] = mp_train
    X_ensemble_train['combined'] = combined_train

    custom_axis = {}
    custom_axis['xlim_min'] = -0.3
    custom_axis['xlim_max'] = 11
    custom_axis['xlim_min'] = -0.3
    custom_axis['ylim_max'] = 11
    custom_axis['ticks'] = np.arange(-0, 6)*2

    display = DisplayData()
    display.alpha = 0.2
    display.markersize = 8
    display.mfc='w'
    display.edgewidth = 0

    svr = SVR(C=150, gamma=0.003)
    gmr = GradientBoostingRegressor(n_estimators=500, max_depth=3)
    rf = RandomForestRegressor(n_estimators=150)
    lr = LinearRegression()

    model = svr

    y_actual, y_predicted, metrics, data_index = cv.cross_validate(X_ensemble_train, y_exp_train, model, N=10, random_state=7, scale_data=False)

#    print('cross validation metrics:', metrics.T.mean())

    # # fit model if suitable results are found
    model.fit(X_ensemble_train, y_exp_train)

    # create ensemble feature vector from model predictions
    X_ensemble_test = pd.DataFrame(index=svr_test.index)
    X_ensemble_test['svr'] = svr_test
#    X_ensemble_test['gbr'] = gbr_test
#    X_ensemble_test['rf'] = rf_test
    #X_ensemble_test['lr'] = lr_test
    #X_ensemble_test['aflow'] = aflow_test
    #X_ensemble_test['mp'] = mp_test
    X_ensemble_test['combined'] = combined_test

    y_ensemble = model.predict(X_ensemble_test)

    test_r2, test_rmse, test_mape = return_metrics(y_exp_test, y_ensemble, X_ensemble_test)
    return test_r2, test_rmse, test_mape


# %%

scheme1_r2, scheme1_rmse, scheme1_mape = scheme1()
scheme2_r2, scheme2_rmse, scheme2_mape = scheme2()
scheme3_r2, scheme3_rmse, scheme3_mape = scheme3()

print('\nBaseline (SVR) score:\nr2: {}\nrmse: {}\nmape: {}'.format(scheme1_r2[0], scheme1_rmse[0], scheme1_mape[0]))
print('\nScheme 1 score:\nr2: {}\nrmse: {}\nmape: {}'.format(scheme1_r2[1], scheme1_rmse[1], scheme1_mape[1]))
print('\nScheme 2 score:\nr2: {}\nrmse: {}\nmape: {}'.format(scheme2_r2[1], scheme2_rmse[1], scheme2_mape[1]))
print('\nScheme 3 score:\nr2: {}\nrmse: {}\nmape: {}'.format(scheme3_r2[1], scheme3_rmse[1], scheme3_mape[1]))

display.actual_vs_predicted(y_exp_test, y_ensemble, data_label='DFT Ensemble Prediction', main_color='#008080', mfc='#008080', custom_axis=custom_axis, save=False, save_name='DFT Ensemble test Prediction')
display.actual_vs_predicted(y_exp_test, X_ensemble_test['svr'], data_label='SVR Prediction', main_color='#f58020', mfc='#f58020', custom_axis=custom_axis, save=False, save_name='SVR test Prediction')
