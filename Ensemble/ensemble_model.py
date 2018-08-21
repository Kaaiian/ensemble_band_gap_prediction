import pandas as pd

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from MachineLearningFunctions.MSE_ML_functions import CrossValidate
from MachineLearningFunctions.MSE_ML_functions import DisplayData

# create objects from custom code
cv = CrossValidate()
display = DisplayData()
display.alpha = 1
display.markersize = 8
display.mfc='w'
# %%

# read in train and test data
df_exp_train = pd.read_csv('ExperimentalData/df_exp_train.csv')
df_exp_test = pd.read_csv('ExperimentalData/df_exp_test.csv')

y_train = df_exp_train.iloc[:, -1]
y_test = df_exp_test.iloc[:, -1]

# %%
# read in experimental predictions
svr_train = pd.read_csv('ExperimentalModels/SupportVectorRegression/predictions/svr_train.csv')


gbr_train = pd.read_csv('ExperimentalModels/GradientBoostingRegression/predictions/gbr_train.csv')


rf_train = pd.read_csv('ExperimentalModels/RandomForestRegression/predictions/rf_train.csv')


lr_train = pd.read_csv('ExperimentalModels/LinearRegression/predictions/lr_train.csv')

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
X_ensemble_train = pd.DataFrame()
X_ensemble_train['svr'] = svr_train
X_ensemble_train['gbr'] = gbr_train
X_ensemble_train['rf'] = rf_train
X_ensemble_train['lr'] = lr_train
X_ensemble_train['aflow'] = aflow_train
X_ensemble_train['mp'] = mp_train
X_ensemble_train['combined'] = combined_train



# %%


#models = [svr, gbr, rf, lr]
names = ['svr', 'gbr', 'rf', 'lr']

svr = SVR(C=10, gamma=1)
gmr = GradientBoostingRegressor()
rf = RandomForestRegressor()
lr = LinearRegression()

model = svr



y_actual, y_predicted, metrics, data_index = cv.cross_validate(X_exp_train, y_exp_train, model, N=5, random_state=1, scale_data=True)








