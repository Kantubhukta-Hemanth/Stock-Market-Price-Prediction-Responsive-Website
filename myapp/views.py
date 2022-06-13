from turtle import color
from unicodedata import name
from django.shortcuts import redirect, render, HttpResponse
from gltf import load_model
from matplotlib import ticker
from sqlalchemy import values
from .models import *
import json

import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import random
import yfinance as yf
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model

# Create your views here.

def postdata(name):
    start = '2010-01-01'
    end = datetime.datetime.now().strftime('%Y-%m-%d')
    #end = '2019-12-31'
    df = data.DataReader(name, 'yahoo', start, end).reset_index()
    df['Prev Close'] = df.Close.shift(1)
    df['change'] = df[['Close', 'Prev Close']].pct_change()['Close'] * 100
    df['High'] = df['High'].round(decimals=4)
    df['Low'] = df['Low'].round(decimals=4)
    df['Open'] = df['Open'].round(decimals=4)
    df['Close'] = df['Close'].round(decimals=4)
    df['Adj Close'] = df['Adj Close'].round(decimals=4)
    df['Prev Close'] = df['Prev Close'].round(decimals=4)
    df['change'] = df['change'].round(decimals=4)
    path = os.getcwd()+f'\\static\\database\\standard\\{name}.csv'
    df.to_csv(path, index=False)
    

def index(request):
    return render(request, 'index.html')

def update_company_info(request):
    tickers = ['HDFC.NS','TCS.NS','RELIANCE.NS','SBIN.NS','TATAMOTORS.NS']
    data = [yf.Ticker(ticker).info for ticker in tickers]
    stocks = dict(zip(tickers, data))
    df = pd.DataFrame.from_dict(stocks)
    df = df.reset_index()
    print(df)
    path = os.getcwd()+f'\\static\\database\\company_info.csv'
    df.to_csv(path, index=False)
    return redirect('/')

def update(request):
    name = ['HDFC.NS','TCS.NS','RELIANCE.NS','SBIN.NS','TATAMOTORS.NS']
    for x in name:
        postdata(x)
    return redirect('/')

def livedata():
    data = yf.download(tickers='HDFC.NS TCS.NS RELIANCE.NS SBIN.NS TATAMOTORS.NS',
            period='1d', interval='1m', group_by='ticker', threads=True).dropna()
    name = ['HDFC.NS','TCS.NS','RELIANCE.NS','SBIN.NS','TATAMOTORS.NS']
    dic, frame = {}, {}

    for col in list(data.columns):
        if frame.get(col[0]) is None:
            frame[col[0]] = {}
        x = col[1]
        if col[1]=='Adj Close':x = 'Adj_Close'
        frame[col[0]][x] = round(data[col][0], 2)
        
    for i in range(3):
        comp = random.choice(name)
        name.remove(comp)
        dic[comp] = frame[comp]['Close']
    return dic, frame


def market(request):
    dic, frame = livedata()
    return render(request, 'market.html', {'company': dic, 'data' : frame})

def candlestick(df):
    fig = go.Figure(
        data = [
            go.Candlestick(
                x = df['Date'],
                high = df['High'],
                low = df['Low'],
                open = df['Open'],
                close = df['Close'],
                name = 'Candlestick'
            ),
            go.Line(
                x=df['Date'], 
                y=(df['Open']+df['Close'])/2,
                marker_color='blue',
                name = 'Average Buy'
            )
        ]
    )
    fig.update_layout(
        height=600,
        margin=dict(l=50,r=50,b=100,t=100),
        paper_bgcolor="LightSteelBlue",
    )

    candlestick_div = plot(fig, output_type='div')
    return candlestick_div

def linegraph(df):
    fig = px.line(df, x='Date', y='Close')
    fig = fig.to_html()
    return fig

def info(request):
    dic, frame= livedata()
    file = os.getcwd()+f'\\static\\database\\company_info.csv'
    data = pd.read_csv(file)
    try:
        value = request.POST['company']
    except:
        value = 'HDFC.NS'
    try:
        start = request.POST['start']
        end = request.POST['end']
    except:
        start = '2020-01-01'
        end = datetime.datetime.now()

    file = os.getcwd()+f'\\static\\database\\standard\\{value}.csv'
    df = pd.read_csv(file)
    start = datetime.datetime.strptime(start, '%Y-%m-%d')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'] >= start) & (df['Date'] <= end)]

    json_records = df.reset_index().to_json(orient ='records')
    bet_data = []
    bet_data = json.loads(json_records)
    for d in bet_data:
        d['Date'] = datetime.datetime.fromtimestamp(d['Date']/1000.0).strftime('%d-%m-%Y')



    '''file = os.getcwd()+f'\\static\\database\\standard\\{value}.csv'
    data = pd.read_csv(file)
    training_data = pd.DataFrame(data['Close'][0:int(len(data)*0.75)])
    testing_data = pd.DataFrame(data['Close'][int(len(data)*0.75) : int(len(data))])
    scaler = MinMaxScaler(feature_range = (0, 1))
    train_data_array = scaler.fit_transform(training_data)
    test_data_array = scaler.fit_transform(testing_data)
    x_train = []
    y_train = []

    for i in range(100, train_data_array.shape[0]):
        x_train.append(train_data_array[i-100 : i])
        y_train.append(train_data_array[i, 0])
    path = os.getcwd()+f'\\static\\models\\model.h5'
    model = load_model(path)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    model.fit(x_train, y_train, epochs=10)
    last_100_days = training_data.tail(100)
    final_df = last_100_days.append(testing_data, ignore_index=True)
    input_data = scaler.fit_transform(final_df)
    x_test = []
    y_test = []
    print(data.tail())

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_predict = model.predict(x_test)
    scale_factor = 1/scaler.scale_
    y_predict = y_predict * scale_factor
    y_test = y_test * scale_factor
    print(y_predict)
    '''

    data.index = data['index']
    del data['index']
    data = data.to_dict()
    data = data[value]
    return render(request, 'info.html', {
        'company': dic, 
        'value': value, 
        'data': data, 
        'df': bet_data, 
        'candlestick': candlestick(df),
        'figure' : linegraph(df)
        }
    )



'''model = Sequential()

        model.add(LSTM(units=50, activation='relu', return_sequences=True,input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))

        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))

        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(units=1, activation='relu'))
        model.compile(optimizer='adam', loss='mean_squared_error')'''
