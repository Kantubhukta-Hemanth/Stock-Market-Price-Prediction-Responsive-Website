from django.shortcuts import redirect, render
from django.template import RequestContext

from SMP.settings import BASE_DIR
from .models import *
import json
from django.core.files.storage import FileSystemStorage

import plotly.graph_objects as go
import plotly
from plotly.offline import plot
import plotly.express as px

import numpy as np
import pandas as pd
from pandas_datareader import data as web
import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import random
import yfinance as yf
from keras.models import load_model

from ta.trend import MACD
from ta.momentum import rsi
# Create your views here.

def postdata(name):
    start = '2007-01-01'
    end = datetime.datetime.now().strftime('%Y-%m-%d')
    #end = '2019-12-31'
    ########df = data.DataReader(name, 'yahoo', start, end).reset_index()

    yf.pdr_override()

    df = web.get_data_yahoo([name], start=start, end=end)

    df = df.reset_index()
    df['Prev Close'] = df.Close.shift(1)
    df['change'] = df[['Close', 'Prev Close']].pct_change()['Close'] * 100
    df['High'] = df['High'].round(decimals=4)
    df['Low'] = df['Low'].round(decimals=4)
    df['Open'] = df['Open'].round(decimals=4)
    df['Close'] = df['Close'].round(decimals=4)
    df['Adj Close'] = df['Adj Close'].round(decimals=4)
    df['Prev Close'] = df['Prev Close'].round(decimals=4)
    df['change'] = df['change'].round(decimals=4)
    return df


def index(request):
    return render(request, 'index.html')

def update(request):
    name = ['HDFC.NS','TCS.NS','RELIANCE.NS','SBIN.NS','TATAMOTORS.NS']
    for x in name:
        try:
            df = postdata(x)
            path = os.path.join(BASE_DIR, f'static/standard/{x}.csv')
            df.to_csv(path, index=False)
        except:
            continue

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
        if col[1]=='Adj Close':
            x = 'Adj_Close'
        frame[col[0]][x] = round(data[col][-1], 2)
        
    for i in range(3):
        comp = random.choice(name)
        name.remove(comp)
        dic[comp] = frame[comp]['Close']
    return dic, frame


def market(request):
    dic, frame = livedata()
    file = os.path.join(BASE_DIR , 'static/Tickers_List.xlsx')

    stocks = pd.read_excel(file)
    names = stocks['Ticker']
    names = json.dumps(names.tolist())

    return render(request, 'market.html', {
        'company': dic,
        'data' : frame,
        'keywords': names
        })

def candlestick(df, value):
    fig = go.Figure(
        data = [
            go.Candlestick(
                x = df['Date'],
                high = df['High'],
                low = df['Low'],
                open = df['Open'],
                close = df['Close'],
                name = value
            )
        ]
    )
    fig.update_layout(
        title=f"{value} stock prices",
        height=600,
        margin=dict(l=50,r=50,b=100,t=100),
        paper_bgcolor="LightSteelBlue",
    )

    

    candlestick_div = plot(fig, output_type='div')
    return candlestick_div

def static_linegraph(df):
    fig = px.line(df, x='Date', y='Close')
    fig = fig.to_html()
    return fig

def info(request):
    dic, frame= livedata()
    file = os.path.join(BASE_DIR , 'static/company_info.csv')
    data = pd.read_csv(file)
    try:
        value = request.POST['company']
    except:
        value = 'HDFC.NS'
    try:
        start = request.POST['start']
        end = request.POST['end']
    except:
        start = '2022-01-01'
        end = datetime.datetime.now()

    file = os.path.join(BASE_DIR, f'static/standard/{value}.csv')
    std_df = pd.read_csv(file)
    start = datetime.datetime.strptime(start, '%Y-%m-%d')
    std_df['Date'] = pd.to_datetime(std_df['Date'])
    std_df = std_df[(std_df['Date'] >= start) & (std_df['Date'] <= end)]

    json_records = std_df.reset_index().to_json(orient ='records')
    bet_data = []
    bet_data = json.loads(json_records)
    for d in bet_data:
        d['Date'] = datetime.datetime.fromtimestamp(d['Date']/1000.0).strftime('%d-%m-%Y')

    data.index = data['index']
    del data['index']
    data = data.to_dict()
    data = data[value]

    info = yf.Ticker(value).fast_info
    
    return render(request, 'info.html', {
        'company': dic, 
        'value': value, 
        'static_data': data, 
        'df': bet_data,
        'prices' : info,
        'candlestick': candlestick(std_df, value),
        }
    )

def pred_graph(df, value):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["y_test"], name="Actual Price", mode="lines"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["y_predict"], name="Predicted", mode="lines"))
    fig.update_layout(
        title=f"{value} Predicted Stock Prices", 
        xaxis_title="Date", 
        yaxis_title="Price"
    )

    fig = plot(fig, output_type='div')

    return fig

def live_candlestick(value):
    fig = go.Figure()

    yf.pdr_override()

    df = yf.download(tickers=value, period='1d',interval='1m')

    fig.add_trace(
            go.Candlestick(
                x = df.index ,
                high = df['High'],
                low = df['Low'],
                open = df['Open'],
                close = df['Close'],
                name = value
            )
    )
    fig.update_layout(
        title=f"{value} Stock Prices",
        yaxis_title = 'Price',
        xaxis_title = 'Date',
        height=600,
        margin=dict(l=50,r=50,b=100,t=100),
        paper_bgcolor="LightSteelBlue",
    )

    fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

    #fig.update_layout(xaxis_rangeslider_visible=False)

    candlestick_div = plot(fig, output_type='div')
    return candlestick_div


def live_graph(df, value):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df["rsi"] = rsi(df["Close"], window=14, fillna=False)

    fig = go.Figure()
    fig = plotly.tools.make_subplots(rows=4, cols=1, shared_xaxes=True,
                    vertical_spacing=0.01, 
                    row_heights=[0.5, 0.15, 0.2, 0.15])

    yf.pdr_override()

    fig.add_trace(
            go.Candlestick(
                x = df['Datetime'] ,
                high = df['High'],
                low = df['Low'],
                open = df['Open'],
                close = df['Close'],
                name = value
            )
    )
    # Add 5-day Moving Average Trace
    fig.add_trace(go.Scatter(x=df['Datetime'], 
                            y=df['MA5'], 
                            opacity=0.7, 
                            line=dict(color='blue', width=2), 
                            name='MA 5'))
    # Add 20-day Moving Average Trace
    fig.add_trace(go.Scatter(x=df['Datetime'], 
                            y=df['MA20'], 
                            opacity=0.7, 
                            line=dict(color='orange', width=2), 
                            name='MA 20'))

    #Volume
    colors = ['green' if row['Open'] - row['Close'] >= 0 
            else 'red' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['Datetime'], 
                        y=df['Volume'],
                        marker_color=colors,
                        name = 'Volume'
                        ), row=2, col=1)
    #RSI
    fig.add_trace(go.Scatter(x=df["Datetime"],
                            y=df["rsi"],
                            mode="lines",
                            line=dict(color='blue', width=1),
                            name="RSI",
                            ), row=3, col=1)
    fig.add_hline(y=30, line_width=1, line_color="Black", row=3, col=1)
    fig.add_hline(y=70, line_width=1, line_color="Black", row=3, col=1)
    #MACD
    macd = MACD(close=df['Close'], 
            window_slow=26,
            window_fast=12, 
            window_sign=9)

    colorsM = ['green' if val >= 0 
            else 'red' for val in macd.macd_diff()]
    fig.add_trace(go.Bar(x=df['Datetime'], 
                        y=macd.macd_diff(),
                        marker_color=colorsM,
                        name='MACD Diff'
                        ), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['Datetime'],
                            y=macd.macd(),
                            line=dict(color='black', width=2),
                            name = 'MACD Buy'
                            ), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['Datetime'],
                            y=macd.macd_signal(),
                            line=dict(color='blue', width=1),
                            name = 'MACD Signal'
                            ), row=4, col=1)
    

    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", showgrid=False, row=3, col=1, tickvals=[30, 70])
    fig.update_yaxes(title_text="MACD", showgrid=False, row=4, col=1)
    fig.update_xaxes(title_text="DATE", row=4, col=1)

    fig.update_layout(
        title=f"{value} Stock Prices",
        yaxis_title = 'Price',
        height=1000,
        margin=dict(l=50,r=50,b=100,t=100),
        paper_bgcolor="LightSteelBlue",
    )

    fig.update_layout(xaxis_rangeslider_visible=False)

    candlestick_div = plot(fig, output_type='div')
    return candlestick_div

def predict(request):
    try:
        value = request.POST['company']
    except:
        value = 'HDFC.NS'
    file = os.path.join(BASE_DIR, f'static/standard/{value}.csv')
    df = pd.read_csv(file)
    training_data = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    testing_data = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])
    
    testing_dates = pd.DataFrame(df['Date'][int(len(df)*0.70) : int(len(df))])
    scaler = MinMaxScaler(feature_range = (0, 1))
    train_data_array = scaler.fit_transform(training_data)
    x_train = []
    y_train = []

    for i in range(100, train_data_array.shape[0]):
        x_train.append(train_data_array[i-100 : i])
        y_train.append(train_data_array[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    path = os.path.join(BASE_DIR, f'static/models/{value[0:-3]}.h5')
    model = load_model(path)
    last_100_days = training_data.tail(100)
    final_df = last_100_days.append(testing_data, ignore_index=True)
    input_data = scaler.fit_transform(final_df)
    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predict = model.predict(x_test)
    scale_factor = 1/scaler.scale_
    y_predict = y_predict * scale_factor
    y_test = y_test * scale_factor
    y_predict = list(map(float, y_predict))
    testing_dates['y_test'] = y_test.tolist()
    testing_dates['y_predict'] = y_predict


    dic, _= livedata()

    try:
        start = request.POST['start']
        end = request.POST['end']
    except:
        start = '2020-01-01'
        end = datetime.datetime.now()
    file = os.path.join(BASE_DIR , 'static/company_info.csv')
    data = pd.read_csv(file)
    data.index = data['index']
    del data['index']
    data = data.to_dict()
    data = data[value]

    file = os.path.join(BASE_DIR, f'static/standard/{value}.csv')
    std_df = pd.read_csv(file)
    start = datetime.datetime.strptime(start, '%Y-%m-%d')
    std_df['Date'] = pd.to_datetime(std_df['Date'])
    std_df = std_df[(std_df['Date'] >= start) & (std_df['Date'] <= end)]

    live = yf.download(value, period='1d', interval='1m', threads=True)
    live = live.reset_index()

    info = yf.Ticker(value).fast_info

    file = os.path.join(BASE_DIR , 'static/Tickers_List.xlsx')

    stocks = pd.read_excel(file)
    names = stocks['Ticker']
    names = json.dumps(names.tolist())

    
    return render(request, 'predict.html',{
            'company': dic,
            'static_data': data,
            'value': value, 
            'prices' : info,
            'keywords': names,
            'linegraph': static_linegraph(std_df),
            'pred_graph': pred_graph(testing_dates, value),
            'live': live_candlestick(value)
        }
    )

def live(request):
    
    dic, _= livedata()
    tickers = ['HDFC.NS','TCS.NS','RELIANCE.NS','SBIN.NS','TATAMOTORS.NS']

    try:
        start = request.POST['start']
        end = request.POST['end']
    except:
        start = '2022-01-01'
        end = datetime.datetime.now()

    try:
        value = request.POST['searchInput']
    except:
        try:
            value = request.POST['company']
        except:
            value = 'HDFC.NS'
    finally:
        if 'start' in request.COOKIES and 'end' in request.COOKIES and 'name' in request.COOKIES:
            old_start = request.COOKIES['start'][:10]
            old_end = request.COOKIES['end']
            old_name = request.COOKIES['name']
            if start!=old_start:
                value = old_name
        start = datetime.datetime.strptime(start, '%Y-%m-%d')
        std_df = postdata(value)
        std_df['Datetime'] = pd.to_datetime(std_df['Date'])
        del std_df['Date']
        std_df = std_df[(std_df['Datetime'] >= start) & (std_df['Datetime'] <= end)]
        print(value)
    
    if value in tickers:
        file = os.path.join(BASE_DIR , 'static/company_info.csv')
        data = pd.read_csv(file)
        data.index = data['index']
        del data['index']
        data = data.to_dict()
        data = data[value]
    else:
        data = False


    info = yf.Ticker(value).fast_info
    currency = info['currency']
    
    if currency=='USD': symbol = '$'
    if currency=='INR': symbol = '₹'

    file = os.path.join(BASE_DIR , 'static/Tickers_List.xlsx')

    stocks = pd.read_excel(file)
    names = stocks['Ticker']
    names = json.dumps(names.tolist())
    
    response = render(request, 'live.html',{
            'company': dic,
            'static_data': data,
            'value': value,
            'keywords': names,
            'prices' : info,
            'symbol' : symbol,
            'linegraph': live_graph(std_df, value),
            'live': live_candlestick(value)
        }
    )

    response.set_cookie(key='start', value=start)
    response.set_cookie(key='end', value=end)
    response.set_cookie(key='name', value=value)

    return response