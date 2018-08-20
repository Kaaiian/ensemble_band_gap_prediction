#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cutsom class to allow for modeling of dft data using Neural Network

"""


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers, regularizers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
import time
import numpy as np
import pandas as pd
from keras.models import model_from_json
from sklearn.externals import joblib


# %%

class ModelDFT:
    '''
    Methods:
---------
fit:
    fit neural network, can choose to evaluate performance while fitting for faster runtime and error estimation at the expense of less accurate model
predict:
    predict data with unscaled feature vector to get new prediction of DFT values
save_model:
    saves the model for reuse later. Saves architeture, weights, and scaling/normalization functions
load_model:
    loads save model
    '''
    def __init__(self):
        self.seed = 100
        np.random.seed(self.seed)
        self.n1 = 1400
        self.drop1 = 0.4
        self.n2 = 800
        self.drop2 = 0.35
        self.n3 = 89
        self.drop3 = 0.2
        self.lr = 0.0005
        self.decay = 5e-5
        pass

    def pre_fit_(self, df, prop, database, epochs=500, batch_size=2500, evaluate=False):
        self.df = df
        self.prop = prop
        self.database = database
        self.get_scaled_X_y_()
        self.N_features = len(self.X_train_columns)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, df, prop, database, epochs=500, batch_size=2500, evaluate=False, deep=False):
        '''
        Parameters
        -----------
        df: pandas.DataFrame object
            dataframe of form Xy, with rows representing a data instance
        prop: str
            the current DFT property being modeled
        database: str
            database from which ```prop``` was taken, options ('aflow', 'mp')
        '''
        self.pre_fit_(df, prop, database, epochs=epochs, batch_size=batch_size, evaluate=evaluate)

        if evaluate is True:
            self.model_fit_metrics_()
        elif deep is True:
            self.deep_fit()
        else:
            self.model_fit_()

    def get_scaled_X_y_(self):
        # get features and target from dataframe
        self.X_train = self.df.iloc[:,:-1]
        self.y_train = self.df.iloc[:,-1]
        self.X_train_index = self.X_train.index.values
        self.X_train_columns = self.X_train.columns.values

        # shuffle the data
        self.X_train = self.X_train.sample(frac=1, random_state=1)
        # reorganize y_train to match the now shuffled X_train dataframe
        self.y_train = self.y_train[self.X_train.index]

        # apply_train scaling
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.normalizer = Normalizer()
        self.X_train = pd.DataFrame(self.normalizer.fit_transform(self.X_train), index=self.X_train_index, columns=self.X_train_columns)

    def scale_X_test_(self):
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.X_test_scaled = self.normalizer.transform(self.X_test_scaled)

    def predict(self, X_test):
        '''
        Parameters
        -----------
        X_test: pd.DataFrame
            unscaled feature vector from which to generate predictions

        Return
        -----------
        y_predictions: pd.Series
            series containing prediction on shared index for each instance in X_test
        '''

        self.X_test = X_test
        self.scale_X_test_()
        self.y_prediction = pd.DataFrame(self.model.predict(self.X_test_scaled), index=self.X_test.index.values, columns=['target prediction'])
#        self.y_prediction = self.model.predict(self.X_test_scaled)
        return self.y_prediction

    def save_model(self):
        path = self.database+'-models/'
        # save scaling
        scaler_filename = path + "scaler " + self.prop +".save"
        joblib.dump(self.scaler, scaler_filename)
        normalizer_filename = path + "normalizer " + self.prop +".save"
        joblib.dump(self.normalizer, normalizer_filename)

        # serialize weights to HDF5
        model_json = self.model.to_json()
        with open(path + "model " + self.prop + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(path + "model "+self.prop+".h5")
        print("Saved model to disk")

    def load_model(self, prop='Band Gap'):
        path = self.database+'-models/'

        # load json and create model
        json_file = open(path + "model " + self.prop + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(path + "model " + self.prop +".h5")
        self.scaler = joblib.load(path + "scaler " + self.prop +".save")
        self.normalizer = joblib.load(path + "normalizer " + self.prop +".save")
        print("Loaded model from disk")
        
    def model_fit_(self):
        t_i = time.time()
        # select the model architecture
        self.model = Sequential()
        self.model.add(Dense(self.n1, input_dim=self.N_features, kernel_regularizer=regularizers.l2(0.00000), kernel_initializer='normal', activation='relu'))
        self.model.add(Dropout(self.drop1))
        self.model.add(Dense(self.n2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0), activation='relu'))
        self.model.add(Dropout(self.drop2))
        self.model.add(Dense(self.n3, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0), activation='relu'))
        self.model.add(Dropout(self.drop3))
        self.model.add(Dense(1, kernel_initializer='normal'))

        # Select learning rate and learning decay
        adm = optimizers.Adam(lr=self.lr, decay=self.decay)

        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer=adm)

        # Fit the model
        self.model.fit(self.X_train, self.y_train,  epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        t_f = time.time()
        print('time:', str(t_f - t_i)+' secs')

    def dummy_model(self, df, prop, database, epochs=500, batch_size=2500, evaluate=False):

        self.pre_fit_(df, prop, database, epochs=epochs, batch_size=batch_size, evaluate=evaluate)

        self.model = Sequential()
        self.model.add(Dense(self.n1, input_dim=self.N_features, kernel_regularizer=regularizers.l2(0.00000), kernel_initializer='normal', activation='relu'))
        self.model.add(Dropout(self.drop1))
        self.model.add(Dense(self.n2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0), activation='relu'))
        self.model.add(Dropout(self.drop2))
        self.model.add(Dense(self.n3, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0), activation='relu'))
        self.model.add(Dropout(self.drop3))
        self.model.add(Dense(1, kernel_initializer='normal'))

        # Select learning rate and learning decay
        adm = optimizers.Adam(lr=self.lr, decay=self.decay)

        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer=adm)


    def deep_fit(self):
        t_i = time.time()
        
        # select the model architecture
        self.model = Sequential()
        self.model.add(Dense(1024, input_dim=self.N_features, kernel_regularizer=regularizers.l2(0.000), kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(1024, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.000), activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(512, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.000), activation='relu'))
        self.model.add(Dense(512, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.000), activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(128, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.000), activation='relu'))
        self.model.add(Dense(128, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.000), activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1, kernel_initializer='normal'))

        # Select learning rate and learning decay
        adm = optimizers.Adam(lr=self.lr, decay=self.decay)

        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer=adm)

        # Fit the model
        history = self.model.fit(self.X_train, self.y_train, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        plt.figure(1, figsize=(7, 7))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot([-0, self.epochs], [0.6, 0.6])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.ylim([0.0, 2*min(history.history['val_loss'])])
        plt.show()
        self.min_mse = min(history.history['val_loss'])
        print('min_err:', self.min_mse)
        self.mse = sum(history.history['val_loss'][-10:])/10
        print('MSE:', self.mse)
        t_f = time.time()
        print('time:', str(t_f - t_i)+' secs')
        
        
    def model_fit_metrics_(self):
        t_i = time.time()

        # select the model architecture
        self.model = Sequential()
        self.model.add(Dense(self.n1, input_dim=self.N_features, kernel_regularizer=regularizers.l2(0.00000), kernel_initializer='normal', activation='relu'))
        self.model.add(Dropout(self.drop1))
        self.model.add(Dense(self.n2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0), activation='relu'))
        self.model.add(Dropout(self.drop2))
        self.model.add(Dense(self.n3, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0), activation='relu'))
        self.model.add(Dropout(self.drop3))
        self.model.add(Dense(1, kernel_initializer='normal'))

        # Select learning rate and learning decay
        adm = optimizers.Adam(lr=self.lr, decay=self.decay)

        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer=adm)

        # Fit the model
        history = self.model.fit(self.X_train, self.y_train, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        plt.figure(1, figsize=(7, 7))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot([-0,1000], [0.5, 0.5])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.ylim([0.0, 2*min(history.history['val_loss'])])
        plt.show()
        self.min_mse = min(history.history['val_loss'])
        print('min_err:', self.min_mse)
        self.mse = sum(history.history['val_loss'][-10:])/10
        print('MSE:', self.mse)
        t_f = time.time()
        print('time:', str(t_f - t_i)+' secs')

