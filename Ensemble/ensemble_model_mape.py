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
display = DisplayData()
display.alpha = 1
display.markersize = 8
display.mfc='w'
# %%

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
    
    scores = [r2_score(y_true, X_ensemble['svr']),
            r2_score(y_true, X_ensemble['gbr']),
            r2_score(y_true, X_ensemble['rf']),
            r2_score(y_true, y_pred)]
    
    rmse = [np.sqrt(mean_squared_error(y_true, X_ensemble['svr'])),
            np.sqrt(mean_squared_error(y_true, X_ensemble['gbr'])),
            np.sqrt(mean_squared_error(y_true, X_ensemble['rf'])),
            np.sqrt(mean_squared_error(y_true, y_pred))]
    
    mape = [calc_mape(y_true, X_ensemble['svr']),
            calc_mape(y_true, X_ensemble['gbr']),
            calc_mape(y_true, X_ensemble['rf']),
            calc_mape(y_true, y_pred)]
    
    print(scores)
    print(rmse)
    print(mape)
    
    print('% score', (scores[0]-scores[-1])/scores[0]*100)
    print('% rmse', (rmse[0]-rmse[-1])/rmse[0]*100)
    print('% mape', (mape[0]-mape[-1])/mape[0]*100)

# %%

# read in train and test data
df_exp_train = pd.read_csv('ExperimentalData/df_exp_train.csv')
df_exp_test = pd.read_csv('ExperimentalData/df_exp_test.csv')

y_exp_train = df_exp_train.iloc[:, -1]
y_exp_test = df_exp_test.iloc[:, -1]


# %%
# read in experimental predictions
svr_train = pd.read_csv('ExperimentalModels/SupportVectorRegression/predictions/svr_train.csv', header=None)
svr_test = pd.read_csv('ExperimentalModels/SupportVectorRegression/predictions/svr_test.csv', header=None)

gbr_train = pd.read_csv('ExperimentalModels/GradientBoostingRegression/predictions/gbr_train.csv', header=None)
gbr_test = pd.read_csv('ExperimentalModels/GradientBoostingRegression/predictions/gbr_test.csv', header=None)

rf_train = pd.read_csv('ExperimentalModels/RandomForestRegression/predictions/rf_train.csv', header=None)
rf_test = pd.read_csv('ExperimentalModels/RandomForestRegression/predictions/rf_test.csv', header=None)

lr_train = pd.read_csv('ExperimentalModels/LinearRegression/predictions/lr_train.csv', header=None)
lr_test = pd.read_csv('ExperimentalModels/LinearRegression/predictions/lr_test.csv', header=None)

# %%
# read in DFT predictions'
aflow_train = pd.read_csv('NeuralNetwork/predictions/train/y_exp_train_predicted NN aflow Band Gap.csv')
aflow_test = pd.read_csv('NeuralNetwork/predictions/test/y_exp_test_predicted NN aflow Band Gap.csv')

mp_train = pd.read_csv('NeuralNetwork/predictions/train/y_exp_train_predicted NN mp Band Gap.csv')
mp_test = pd.read_csv('NeuralNetwork/predictions/test/y_exp_test_predicted NN mp Band Gap.csv')

combined_train = pd.read_csv('NeuralNetwork/predictions/train/y_exp_train_predicted NN combined Band Gap.csv')
combined_test = pd.read_csv('NeuralNetwork/predictions/test/y_exp_test_predicted NN combined Band Gap.csv')

# %%

# create ensemble feature vector from model predictions
X_ensemble_train = pd.DataFrame(index=svr_train.index)
X_ensemble_train['svr'] = svr_train
X_ensemble_train['gbr'] = gbr_train
X_ensemble_train['rf'] = rf_train
#X_ensemble_train['lr'] = lr_train
#X_ensemble_train['aflow'] = aflow_train
#X_ensemble_train['mp'] = mp_train
X_ensemble_train['combined'] = combined_train

#display.actual_vs_predicted(X_ensemble_train['combined'], X_ensemble_train['aflow'])
#display.actual_vs_predicted(y_exp_train, X_ensemble_train['svr'])


#models = [svr, gbr, rf, lr]
#names = ['svr', 'gbr', 'rf', 'lr']

#svr = SVR(C=100, gamma=0.001)
svr = SVR(C=150, gamma=0.003)
gmr = GradientBoostingRegressor(n_estimators=500, max_depth=3)
rf = RandomForestRegressor(n_estimators=150)
lr = LinearRegression()
#
model = svr
#
#y_actual, y_predicted, metrics, data_index = cv.cross_validate(X_ensemble_train, y_exp_train, model, N=10, random_state=7, scale_data=False)
#display.actual_vs_predicted(y_actual, y_predicted)
#print(metrics.T.mean())
#print((y_actual, y_predicted))

# %%
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

return_metrics(y_exp_test, y_ensemble, X_ensemble_test)

display.actual_vs_predicted(y_exp_test, y_ensemble)
display.actual_vs_predicted(y_exp_test, X_ensemble_test['svr'])
