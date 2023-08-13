# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 22:04:11 2022

@author: ChakalasiyaMayurVash
"""

## Imports libs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
print(tf.__version__) 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import itertools
import random
import os
import math # Mathematical functions 
import time

ts = str(time.time())

def load_time_series_data_and_preprocessing(filename,f_date,t_date):
    
    #Load data into dataframe
    data =  pd.read_csv(filename, header=0)
    
    #information about dataset
    print(data.info())
    
    #missing = data[col_name].isnull()
    #data[missing]
    
    #Duplicate 
    
    print("Missing values:")
    print(data.isnull().sum())
    
    #Convert Date from Object to Datetime type
    data['Date'] = pd.to_datetime(data['Date'])
    
    #Sorting value by Date
    data=data.sort_values(by="Date", ascending=True)
    
    #Setting index as Date
    data.set_index("Date",inplace=True)
    
    #Dataset Selection by date from dataset
    data=data[(data.index >= f_date) & (data.index <= t_date)]
    
    return data
    

def feature_selection_scaling(data):
    data_prices = data.drop(['SMAVG_100d'], axis=1)
    # We add a prediction column and set dummy values to prepare the data for scaling
    data_prices_ext = data_prices.copy()
    data_prices_ext['Prediction'] = data_prices_ext['Close']  
    
    # Get the number of rows in the data
    nrows = data_prices.shape[0]
    
    # Convert the data to numpy values
    np_data_unscaled = np.array(data_prices)
    np_data = np.reshape(np_data_unscaled, (nrows, -1))
    print('np_data.shape:',np_data.shape)    
    
    # Transform the data by scaling each feature to a range between 0 and 1
    scaler = MinMaxScaler()
    np_data_scaled = scaler.fit_transform(np_data_unscaled)
    
    # Creating a separate scaler that works on a single column for scaling predictions
    scaler_pred = MinMaxScaler()
    df_Close = pd.DataFrame(data_prices_ext['Close'])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)
    
    # Print the tail of the dataframe
    return data_prices, data_prices_ext,np_Close_scaled,np_data_scaled,scaler_pred
    

# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, sequence_length time steps per sample, and 6 features
def partition_dataset(sequence_length, data, index_Close):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction
    
    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

def transform_multivariate_data(data,np_data_scaled,sequence_length):
    # Set the sequence length - this is the timeframe used to make a single prediction
    #sequence_length = 6
    
    # Prediction Index
    index_Close = data.columns.get_loc("Close")
    
    # Split the training data into train and train data sets
    # As a first step, we get the number of rows to train the model on 80% of the data 
    train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)
    
    # Create the training and test data
    train_data = np_data_scaled[0:train_data_len, :]
    test_data = np_data_scaled[train_data_len - sequence_length:, :]
    
    # Generate training data and test data
    x_train, y_train = partition_dataset(sequence_length, train_data,index_Close)
    x_test, y_test = partition_dataset(sequence_length, test_data,index_Close)

    # Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
    print('x_train.shape:',x_train.shape,'y_train.shape:', y_train.shape)
    print('x_test.shape:',x_test.shape,'y_test.shape:', y_test.shape)
        
    # Validate that the prediction value and the input match up
    # The last close price of the second input sample should equal the first prediction value
    print(x_train[1][sequence_length-1][index_Close])
    print(y_train[0])
    return  x_train, y_train, x_test, y_test, train_data_len


def evalute_model_performance(history,output):
    output= output+ "evalute_model_performance" + ts + ".png"
    fig = plt.figure(figsize=(20,7))
    fig.add_subplot(121)
    
    # Accuracy
    plt.plot(history.epoch, history.history['root_mean_squared_error'], label = "rmse")
    plt.plot(history.epoch, history.history['val_root_mean_squared_error'], label = "val_rmse")
    
    plt.title("RMSE", fontsize=18)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("RMSE", fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend()
    
    
    #Adding Subplot 1 (For Loss)
    fig.add_subplot(122)
    
    plt.plot(history.epoch, history.history['loss'], label="loss")
    plt.plot(history.epoch, history.history['val_loss'], label="val_loss")
    
    plt.title("Loss", fontsize=18)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
    plt.savefig(output)

def train_multivariate_prediction_model(x_train, y_train,filename):
    # ------------------LSTM-----------------------
    regressor = Sequential()
    regressor.add(LSTM(units=16, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units=16, return_sequences=False))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1, activation='linear'))
    regressor.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    print('LSTM Regression Summary :')
    print(regressor.summary())
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    #mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    
    # fit model
    history = regressor.fit(x_train, y_train, validation_split=0.3, epochs=40, batch_size=64, callbacks=[es])
    # plot Accuracy and Loss 
    evalute_model_performance(history,filename)
    
    results = regressor.evaluate(x_test, y_test)
    print("test loss, test acc:", np.round(results, 4))
    return history, results

def tune_model_hyperparam(config, x_train, y_train, x_test, y_test):
    
    first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout = config
    
    possible_combinations = list(itertools.product(first_additional_layer, second_additional_layer, third_additional_layer,
                                                  n_neurons, n_batch_size, dropout))
    
    print(possible_combinations)
    print('\n')
    
    hist = []
    
    for i in range(0, len(possible_combinations)):
        
        print(f'{i+1}th combination: \n')
        print('--------------------------------------------------------------------')
        
        first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout = possible_combinations[i]
        
        # instantiating the model in the strategy scope creates the model on the TPU
        #with tpu_strategy.scope():
        regressor = Sequential()
        regressor.add(LSTM(units=n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        regressor.add(Dropout(dropout))

        if first_additional_layer:
            regressor.add(LSTM(units=n_neurons, return_sequences=True))
            regressor.add(Dropout(dropout))

        if second_additional_layer:
            regressor.add(LSTM(units=n_neurons, return_sequences=True))
            regressor.add(Dropout(dropout))

        if third_additional_layer:
            regressor.add(GRU(units=n_neurons, return_sequences=True))
            regressor.add(Dropout(dropout))

        regressor.add(LSTM(units=n_neurons, return_sequences=False))
        regressor.add(Dropout(dropout))
        regressor.add(Dense(units=1, activation='linear'))
        regressor.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        '''''
        From the mentioned article above --> If a validation dataset is specified to the fit() function via the validation_data or v
        alidation_split arguments,then the loss on the validation dataset will be made available via the name “val_loss.”
        '''''

        file_path = 'best_model.h5'

        mc = ModelCheckpoint(file_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        '''''
        cb = Callback(...)  # First, callbacks must be instantiated.
        cb_list = [cb, ...]  # Then, one or more callbacks that you intend to use must be added to a Python list.
        model.fit(..., callbacks=cb_list)  # Finally, the list of callbacks is provided to the callback argument when fitting the model.
        '''''

        regressor.fit(x_train, y_train, validation_split=0.3, epochs=40, batch_size=n_batch_size, callbacks=[es, mc], verbose=0)

        # load the best model
        # regressor = load_model('best_model.h5')

        train_accuracy = regressor.evaluate(x_train, y_train, verbose=0)
        test_accuracy = regressor.evaluate(x_test, y_test, verbose=0)

        hist.append(list((first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout,
                          train_accuracy, test_accuracy)))

        print(f'{str(i)}-th combination = {possible_combinations[i]} \n train accuracy: {train_accuracy} and test accuracy: {test_accuracy}')
        
        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')
        
        hist = pd.DataFrame(hist)
        hist = hist.sort_values(by=[7], ascending=True)
        print(hist)
        print(f'Best Combination: \n first_additional_layer = {hist.iloc[0, 0]}\n second_additional_layer = {hist.iloc[0, 1]}\n third_additional_layer = {hist.iloc[0, 2]}\n n_neurons = {hist.iloc[0, 3]}\n n_batch_size = {hist.iloc[0, 4]}\n dropout = {hist.iloc[0, 5]}')
        print('**************************')
        print(f'Results Before Tunning:\n Test Set RMSE: {np.round(results, 4)[1]}\n')
        print(f'Results After Tunning:\n Test Set RMSE: {np.round(hist.iloc[0, -1], 4)[1]}\n')
        print(f'{np.round((results[1] - hist.iloc[0, -1][1])*100/np.round(results, 4)[1])}% Improvement') 
    return hist

def train_model_with_optimized_hyperparam(first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout):
    regressor = Sequential()
    regressor.add(LSTM(units=n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    regressor.add(Dropout(dropout))
    
    if first_additional_layer:
        regressor.add(LSTM(units=n_neurons, return_sequences=True))
        regressor.add(Dropout(dropout))
    
    if second_additional_layer:
        regressor.add(LSTM(units=n_neurons, return_sequences=True))
        regressor.add(Dropout(dropout))
    
    if third_additional_layer:
        regressor.add(GRU(units=n_neurons, return_sequences=True))
        regressor.add(Dropout(dropout))
    
    regressor.add(LSTM(units=n_neurons, return_sequences=False))
    regressor.add(Dropout(dropout))
    regressor.add(Dense(units=1, activation='linear'))
    regressor.compile(optimizer='adam', loss='mse')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    
    file_path = 'best_model.h5'
    
    mc = ModelCheckpoint(file_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    regressor.fit(x_train, y_train, validation_split=0.3, epochs=40, batch_size=n_batch_size, callbacks=[es, mc], verbose=0)
    
    results=regressor.evaluate(x_test, y_test)
    print("test loss, test acc:", np.round(results, 4))
    return regressor

def predict_future_price(x_test,y_test,train_data_len,from_date,cryto_name,output):
    output = output + 'real_price_pred_price_'+ts+".png"
    y_pred_scaled = regressor.predict(x_test)
    
    plt.figure(figsize=(16,8), dpi= 100, facecolor='w', edgecolor='k')
    
    plt.plot(y_test, color='red', label = 'Real Close Price')
    plt.plot(y_pred_scaled, color='green', label = 'Predicted Close Price')
    plt.legend(loc='best')
    
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))
    
    #display_start_date = "2022-04-01" 

    # Add the difference between the valid and predicted prices
    train = pd.DataFrame(data_prices_ext['Close'][:train_data_len + 1]).rename(columns={'Close': 'y_train'})
    valid = pd.DataFrame(data_prices_ext['Close'][train_data_len:]).rename(columns={'Close': 'y_test'})
    valid.insert(1, "y_pred", y_pred, True)
    valid.insert(1, "residuals", valid["y_pred"] - valid["y_test"], True)
    df_union = pd.concat([train, valid])
    # Zoom in to a closer timeframe
    df_union_zoom = df_union[df_union.index > from_date]
    print(df_union_zoom)
    
    # Create the lineplot
    fig, ax1 = plt.subplots(figsize=(16, 8))
    plt.title("y_pred vs y_test")
    plt.ylabel(cryto_name, fontsize=18)
    sns.set_palette(["#090364", "#1960EF", "#EF5919"])
    sns.lineplot(data=df_union_zoom[['y_pred', 'y_train', 'y_test']], linewidth=1.0, dashes=False, ax=ax1)
    
    # Create the bar plot with the differences
    df_sub = ["#2BC97A" if x > 0 else "#C92B2B" for x in df_union_zoom["residuals"].dropna()]
    ax1.bar(height=df_union_zoom['residuals'].dropna(), x=df_union_zoom['residuals'].dropna().index, width=3, label='residuals', color=df_sub)
    plt.legend()
    plt.show()
    plt.savefig(output)
    
#-------------------------------------------------------------------------------------
filename = "./dataset/bitcoin/bitcoin_30min_dataset.csv"
cryto_name = "bitcoin"
from_date = "2022-04-01"
to_date = "2022-09-30"
output="./dataset/"+cryto_name+"/output"
#-------------------------------------------------------------------------------------

# Set the sequence length - this is the timeframe used to make a single prediction
sequence_length = 25

data=load_time_series_data_and_preprocessing(filename,from_date,to_date)
feature = []
data_prices, data_prices_ext, np_Close_scaled,np_data_scaled,scaler_pred = feature_selection_scaling(data)

x_train, y_train, x_test, y_test,train_data_len = transform_multivariate_data(data,np_data_scaled,sequence_length)

history, results = train_multivariate_prediction_model(x_train, y_train,output)

config = [[False], [False], [False], [16, 32], [8, 16, 32], [0.2]]  

# list of lists --> [[first_additional_layer], [second_additional_layer], [third_additional_layer], [n_neurons], [n_batch_size], [dropout]]
hist = tune_model_hyperparam(config, x_train, y_train, x_test, y_test)  # change x_train shape

first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout = list(hist.iloc[0, :-2])

regressor = train_model_with_optimized_hyperparam(first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout)

predict_future_price(x_test,y_test,train_data_len,from_date,cryto_name,output)