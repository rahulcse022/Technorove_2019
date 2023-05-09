#!/usr/bin/env python
# coding: utf-8

# In[13]:


# import requests
# from datetime import datetime
# import pandas as pd


# def get_filename(from_symbol, to_symbol, exchange, datetime_interval, download_date):
#     return '%s_%s_%s_%s_%s.csv' % (from_symbol, to_symbol, exchange, datetime_interval, download_date)


# def download_data(from_symbol, to_symbol, exchange, datetime_interval):
#     supported_intervals = {'minute', 'hour', 'day'}
#     assert datetime_interval in supported_intervals,\
#         'datetime_interval should be one of %s' % supported_intervals

#     print('Downloading %s trading data for %s %s from %s' %
#           (datetime_interval, from_symbol, to_symbol, exchange))
#     base_url = 'https://min-api.cryptocompare.com/data/histo'
#     url = '%s%s' % (base_url, datetime_interval)

#     params = {'fsym': from_symbol, 'tsym': to_symbol,
#               'limit': 2000, 'aggregate': 1,
#               'e': exchange}
#     request = requests.get(url, params=params)
#     data = request.json()
#     return data


# def convert_to_dataframe(data):
#     df = pd.io.json.json_normalize(data, ['Data'])
#     df['datetime'] = pd.to_datetime(df.time, unit='s')
#     df = df[['datetime', 'low', 'high', 'open',
#              'close', 'volumefrom', 'volumeto']]
#     return df


# def filter_empty_datapoints(df):
#     indices = df[df.sum(axis=1) == 0].index
#     print('Filtering %d empty datapoints' % indices.shape[0])
#     df = df.drop(indices)
#     return df

# from_symbol = 'BTC'
# to_symbol = 'USD'
# exchange = 'Bitstamp'
# datetime_interval = 'day'
# data = download_data(from_symbol, to_symbol, exchange, datetime_interval)
# df = convert_to_dataframe(data)
# df = filter_empty_datapoints(df)

# current_datetime = datetime.now().date().isoformat()
# filename = get_filename(from_symbol, to_symbol, exchange, datetime_interval, current_datetime)
# print('Saving data to %s' % filename)


# In[14]:


# df.tail()


# In[3]:


import requests
from datetime import datetime
import pandas as pd



def download_data(from_symbol, to_symbol, exchange, datetime_interval):
    supported_intervals = {'minute', 'hour', 'day'}
    assert datetime_interval in supported_intervals,        'datetime_interval should be one of %s' % supported_intervals

    print('Downloading %s trading data for %s %s from %s' %
          (datetime_interval, from_symbol, to_symbol, exchange))
    base_url = 'https://min-api.cryptocompare.com/data/histo'
    url = '%s%s' % (base_url, datetime_interval)

    params = {'fsym': from_symbol, 'tsym': to_symbol,
              'limit': 2000, 'aggregate': 1,
              'e': exchange}
    request = requests.get(url, params=params)
    data = request.json()
    return data


def convert_to_dataframe(data):
    df = pd.io.json.json_normalize(data, ['Data'])
    df['datetime'] = pd.to_datetime(df.time, unit='s')
    df = df[['datetime', 'low', 'high', 'open',
             'close', 'volumefrom', 'volumeto']]
    return df


def filter_empty_datapoints(df):
    indices = df[df.sum(axis=1) == 0].index
    print('Filtering %d empty datapoints' % indices.shape[0])
    df = df.drop(indices)
    return df

from_symbol = 'BTC'
to_symbol = 'USD'
exchange = 'Bitstamp'
datetime_interval = 'day'
data = download_data(from_symbol, to_symbol, exchange, datetime_interval)
df = convert_to_dataframe(data)
df = filter_empty_datapoints(df)

current_datetime = datetime.now().date().isoformat()
df.index = df['datetime']


# In[4]:


# df.head()


# In[5]:





# In[12]:


# df.head()


# In[11]:


# df.tail()


# In[8]:


# Bitcoin data prediction tool
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime, timedelta
import plotly          #(version 4.4.1)
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
# df = pd.read_csv('BTC-USD.csv')
time_step = 5

btc_model = load_model("BTC_Model.h5")

df_d = np.array(df['close']).reshape(-1,1)


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df_d=scaler.fit_transform(np.array(df_d).reshape(-1,1))


##splitting dataset into train and test split
training_size=int(len(df_d)*0.75)
test_size=len(df_d)-training_size
train_data,test_data=df_d[0:training_size,:],df_d[training_size:len(df_d),:1]

x_input=test_data[len(test_data)-time_step:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=5
i=0
while(i<10):
    
    if(len(temp_input)>5):
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = btc_model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = btc_model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    

predicted_10_days = scaler.inverse_transform(lst_output)
predicted_10_days = list(predicted_10_days[:,0])

btc_index = df['datetime']
btc_data = df['close']


last_day_from_dataset = btc_index[len(btc_index)-1:]
last_day_from_dataset = pd.to_datetime(last_day_from_dataset, format="%Y-%m-%d")


future_10_days = pd.date_range(start=pd.to_datetime(last_day_from_dataset.value_counts().index[0], format="%Y-%m-%d"), periods=11, freq='D')

future_10_days = future_10_days[1:]

future_10_days = pd.to_datetime(future_10_days, format="%Y-%m-%d")

future_10_days = pd.DataFrame(future_10_days)[0]


predict_days_index = []
for x in future_10_days:
    day = f"{x.year}-{x.month}-{x.day}"
    predict_days_index.append(day)

prediction_10 = pd.DataFrame(predicted_10_days, index=future_10_days, columns=['pred'])

btc_data_2 = prediction_10['pred']
btc_index_2 = prediction_10.index


btc_index_2 = btc_index.copy()


pred_btc = btc_data_2.reset_index()

btc_index_2 = btc_index_2.append(pred_btc[0], ignore_index=True)
btc_data_2 = btc_data_2.append(pred_btc['pred'], ignore_index=True)

mark_values = {
    2015:'2015',
    2016:'2016',
    2017:'2017',
    2018:'2018',
    2019:'2019',
    2020:'2020',
    2021:'2021',
    2022:'2022'
}
df['date'] = pd.to_datetime(df['datetime'])
df['year'] = df['date'].apply(lambda x: x.year)

colors = {
   'background': '#00ff99',
   'text': '#ff0033'
}
print("Program Started ...............................................")
data_len=150


app.layout = html.Div( children=[
   
    html.H1("Technorove Bitcoin Price Prediction & Forcasting System", style={"textAlign": "center", 'backgroundColor': colors['background'] }),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='BitCoin Actual Data',children=[
            html.Div([
                html.H1("BitCoin Actual Closing Price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x = btc_index,
                                y = btc_data,
                                mode='lines+markers',
                            )

                        ],
                        "layout":go.Layout(
                            title='Closing Rate vs Date',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'},
                            
                        ),
                    })
            ]) ,
            ### new Graph from here


      


        ]),
        dcc.Tab(label='BitCoin Price Actual + Prediction ', children=[
            html.Div([
                html.H1("BitCoin Actual + Predictions", 
                        style={'textAlign': 'center'}),
                
                dcc.Graph(
                    id="Predicted Data",
                
                    figure={
                        
                        
                        "data":[
                            go.Scatter(
                                x = btc_index[-data_len:],
                                y = btc_data[-data_len:],
                                mode='lines+markers',
                                name = "Actual Data"
                            ),
                            
                            go.Scatter(
                                x = predict_days_index,
                                y = predicted_10_days,
                                mode='lines+markers',
                                name = "Predicted Data"
                            ),
                           
                        ],
                        
                        "layout":go.Layout(
                            title='Closing Rate vs Date',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'},

                        )
                    }

                ),
  
            ], className="container"),
              
            
        ]),
        # Third Tab from here
        dcc.Tab(label='BitCoin Actual Data',children=[
            
                    html.Div([
            html.H1(children= "BitCoin Actual Closing Price",style={"text-align": "center", "color":"black"})
        ]),

        html.Div([
            dcc.Graph(id='the_graph')
        ]),

        html.Div([
            dcc.RangeSlider(id='the_year',
                min=2016,
                max=2021,
                value=[2020,2021],
                marks=mark_values,
                step=None)
        ],style={"width": "70%", "position":"absolute",
                 "left":"5%"})

      


        ]) # 3rd tab end ....!

    ])
])


print(pd.DataFrame(data=predict_days_index,index=predicted_10_days))



@app.callback(
    Output('the_graph','figure'),
    [Input('the_year','value')]
)
def update_graph(years_chosen):
    dff=df[(df['year']>=years_chosen[0])&(df['year']<=years_chosen[1])]
    scatterplot = px.line(
        data_frame=dff,
        x="datetime",
        y="close",
        height=550
    )
    scatterplot.update_traces(textposition='top center')
    return (scatterplot)


# In[9]:


if __name__=='__main__':
    app.run_server(debug=False, port="8000")

