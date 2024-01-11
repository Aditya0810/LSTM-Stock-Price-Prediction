import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import streamlit as st
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.title("Forecast Markets with Time Series Analysis Models")
st.info("This project is based on the study 'Stock Price Prediction using LSTM and ARIMA' by IEEE. Utilising the two of some of the most valid models for time series forecasting in such contexts according to this study.", icon="ℹ️")
drop_down0 = st.selectbox("Which model would you like to try:", ('LSTM', 'ARIMA'))
drop_down1 = st.selectbox("Select the epoch size, (A higher epoch size will result in a longer estimated time)", ('50','100','500'))
stock_symbol = st.text_input("Enter the symbol of the stock")
j = int(st.slider("How many days in the future would you like to forecast", min_value=1, max_value=150, value=1, step=1, key="day_input"))
col0, col1 = st.columns(2)
with col0:
    clicked1 = st.button("Forecast Stock Prices")

def lstm():
    data = yf.download(tickers=stock_symbol,period='5y')
    close = data[['Close']]
    df = close.values
    df_reshaped = df.reshape(1,-1)
    normalizer = MinMaxScaler(feature_range=(0,1))

    df_scaled = normalizer.fit_transform(np.array(df).reshape(-1,1))
    train_size = int(len(df_scaled)*0.70)
    test_size = len(df_scaled) - train_size
    df_train, df_test = df_scaled[0:train_size,:], df_scaled[train_size:len(df_scaled),:1]
    df_test_rows, df_test_columns = df_test.shape
    def create_df(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

    time_stamp = 100
    X_train, y_train = create_df(df_train,time_stamp)
    X_test, y_test = create_df(df_test,time_stamp)

    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1,activation='linear'))

    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=int(drop_down1),batch_size=64)

   
    loss = model.history.history['loss']

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)

    fut_inp = df_test[((df_test_rows)-100):]
    fut_inp = fut_inp.reshape(1,-1)
    tmp_inp = list(fut_inp)
    tmp_inp = tmp_inp[0].tolist()
    lst_output=[]
    n_steps=100
    i=0
    while(i<j):
        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape(1, n_steps, 1)
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat[0].tolist())
            i += 1
        else:
            fut_inp = fut_inp.reshape(1, n_steps,1)
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat[0].tolist())
            i += 1

    output_array = np.array(lst_output).reshape(-1,1)
    rescaled_predictions = normalizer.inverse_transform(output_array)
    with col0:
        st.write("The predicted prices are")
        st.line_chart(rescaled_predictions, use_container_width=True)

def arima():
    data = yf.download(tickers=stock_symbol,period='5y')
    close = data[['Close']]
    df = close.values
    df_reshaped = df.reshape(1,-1)
    normalizer = MinMaxScaler(feature_range=(0,1))

    df_scaled = normalizer.fit_transform(np.array(df).reshape(-1,1))
    train_size = int(len(df_scaled)*0.70)
    test_size = len(df_scaled) - train_size
    df_train, df_test = df_scaled[0:train_size,:], df_scaled[train_size:len(df_scaled),:1]
    df_test_rows, df_test_columns = df_test.shape
    def create_df(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

    time_stamp = 200
    X_train, y_train = create_df(df_train,time_stamp)
    X_test, y_test = create_df(df_test,time_stamp)
    stepwise_fit = auto_arima(y_train, suppress_warnings=True)
    order1 = stepwise_fit.get_params()['order']
    model = ARIMA(y_train, order = order1)
    model_fit = model.fit()
    pred = model_fit.predict(start=len(y_train), end = len(y_train)+j-1, type='levels')
    pred_arr = np.array(pred).reshape(-1,1)
    pred_original_scale = normalizer.inverse_transform(pred_arr)
    with col0:
        st.write("The predicted prices are")
        st.line_chart(pred_original_scale, use_container_width=True)

    

if clicked1 and drop_down0 == 'LSTM':
    lstm()
elif clicked1 and drop_down0 == 'ARIMA':
    arima()


