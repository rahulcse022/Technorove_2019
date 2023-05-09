# Bitcoin data prediction tool
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime, timedelta
import plotly          #(version 4.4.1)
import plotly.express as px
import json
import requests
import plotly.graph_objs as go
import dash_table


all_coin_list = ['BTC','EOS','ETH','LTC','XRP','BCH', "ADA", "WOZX", "XLM", "DOGE", "DOT"]
# all_coin_list = ['BTC','EOS','ETH']
Data = dict()
to_symbol    =    "USD"
limit = "1000"



for coin_n in all_coin_list:    
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym={0}&tsym={1}&limit={2}'.format(coin_n, to_symbol, limit))
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    hist['datetime'] = hist.index
    hist.drop(axis=1, columns=['volumefrom','volumeto','conversionType','conversionSymbol'], inplace=True)
    target_col = 'close'
    Data.update({coin_n:hist})


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server


btc_model = load_model("BTC_Model.h5")
eos_model = load_model("EOS_Model.h5")
eth_model = load_model("ETH_Model.h5")
ltc_model = load_model("LTC_Model.h5")
xrp_model = load_model("XRP_Model.h5")
bch_model = load_model("BCH_Model.h5")
ada_model = load_model("ADA_Model.h5")
wozx_model = load_model("WOZX_Model.h5")
XLM_model = load_model("XLM_Model.h5")
doge_model = load_model("DOGE_Model.h5")
dot_model = load_model("DOT_Model.h5")



def Prediction_10_days(Data_index, train_test_size, n_steps):
    time_step = n_steps

    df_d = np.array(Data[all_coin_list[Data_index]]['close']).reshape(-1,1)


    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    df_d=scaler.fit_transform(np.array(df_d).reshape(-1,1))


    ##splitting dataset into train and test split
    training_size=int(len(df_d)*train_test_size)
    test_size=len(df_d)-training_size
    train_data,test_data=df_d[0:training_size,:],df_d[training_size:len(df_d),:1]

    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()


    # demonstrate prediction for next 10 days
    from numpy import array

    lst_output=[]
#     n_steps=5
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

    btc_index = Data[all_coin_list[Data_index]]['datetime']
    btc_data = Data[all_coin_list[Data_index]]['close']


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
    
    Data[all_coin_list[Data_index]]['year'] = pd.to_datetime( Data[all_coin_list[1]]['datetime']).apply(lambda x: x.year)
    
    return btc_index_2, btc_data_2



btc_pred_index, btc_pred_data = Prediction_10_days(0, 0.75, 5)
eos_pred_index, eos_pred_data = Prediction_10_days(1, 0.75, 5)
eth_pred_index, eth_pred_data = Prediction_10_days(2, 0.75, 5)
ltc_pred_index, ltc_pred_data = Prediction_10_days(3, 0.75, 5)
xrp_pred_index, xrp_pred_data = Prediction_10_days(4, 0.75, 5)
bch_pred_index, bch_pred_data = Prediction_10_days(5, 0.75, 5)
ada_pred_index, ada_pred_data = Prediction_10_days(6, 0.95, 5)
wozx_pred_index, wozx_pred_data = Prediction_10_days(7, 0.95, 5)
xlm_pred_index, xlm_pred_data = Prediction_10_days(6, 0.95, 5)
doge_pred_index, doge_pred_data = Prediction_10_days(6, 0.95, 5)
dot_pred_index, dot_pred_data = Prediction_10_days(6, 0.95, 5)


BTC = Data['BTC'].copy()
EOS = Data['EOS'].copy()
ETH = Data['ETH'].copy()
LTC = Data['LTC'].copy()
XRP = Data['XRP'].copy()
BCH = Data['BCH'].copy()
ADA = Data['ADA'].copy()
WOZX = Data['WOZX'].copy()
XLM = Data['XLM'].copy()
DOGE = Data['DOGE'].copy()
DOT = Data['DOT'].copy()


# In[4]:



def Last_DataFrame(dataf, pred_data, pred_index):
    close_value = list(dataf['close'].values)
    NAN = [np.nan for x in range(len(close_value))]
    d = dict()
    d.update({
    "Actual":close_value + NAN[:10],
    "Prediction": list(NAN) + list(pred_data[-10:]),
    "Date" : pred_index
    })
    return pd.DataFrame(d)  




btc_pred_index[-10:].values


# In[6]:


def date_convert(x):
    ts = pd.to_datetime(str(btc_pred_index[-10:].values[x])) 
    d = ts.strftime('%d-%m-%Y')
    return d
date_p = []
for i in range(10):
    date_p.append(date_convert(i))
print(date_p)
print(date_p[0])
print(type(date_p[0]))


# In[ ]:





# In[7]:


pred_table_data = pd.DataFrame({
    "Date": date_p,
    "Prediction":btc_pred_data[:10]
})
pred_table_data.head()


## ***************** Bar Chart Start ***************************

# all_coin_list = ['YFI','MKR']
coin_list_under_5k = ['MKR','BNB','COMP','KSM','AAVE','XMR','BSV','ZEC','EGLD','DCR','FIL','BTG','ZEN']
print(len(coin_list_under_5k))
Data_5k = dict()
to_symbol    =    "USD"
limit = "1000"


for coin_n in coin_list_under_5k:    
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym={0}&tsym={1}&limit={2}'.format(coin_n, to_symbol, limit))
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    hist['datetime'] = hist.index
    hist.drop(axis=1, columns=['volumefrom','volumeto','conversionType','conversionSymbol'], inplace=True)
    target_col = 'close'
    Data_5k.update({coin_n:hist})

for i,j in Data_5k.items():
    print(i, len(Data_5k[i]))


last_day_data = []
for i in Data_5k.keys():
    last_day_data.append(Data_5k[i].tail(1)['close'].values[0])

print(last_day_data)
print(coin_list_under_5k)

bar_data = pd.DataFrame({
    "Price" : last_day_data,
    "Cryptocurrency" : coin_list_under_5k
    })


# df = px.data.iris()
fig_bar = px.bar(bar_data, y="Price", x="Cryptocurrency")
fig_bar.update_traces(text=bar_data["Price"],texttemplate='%{text:.2s}', textposition='outside')
fig_bar.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')


### ******************* Bar Chart End ***************************

app.layout = html.Div( children=[

            html.H2("Technorove Cryprocurrency Price Prediction & Forcasting System",
                style={
                    "textAlign": "center",
                    'backgroundColor': "#00ff99"
                     }
                     ),

            # html.Div([

        # html.Br(),
        # html.Div(id='output_data'),
        html.Br(),

        dcc.Dropdown(id='my_dropdown',
            options=[
                     {'label': 'ETherem (ETH)', 'value': 'ETH'},
                     {'label': 'Bitcoin (BTC)', 'value': 'BTC'},
                     {'label': 'XRP', 'value': 'XRP'},
                     {'label': 'Bitcoin Cash (BCH)', 'value': 'BCH'},
                     {'label': 'Litecoin (LTC)', 'value': 'LTC'},
                     {'label': 'EOS (EOS)', 'value': 'EOS'},
                    {'label': 'Cardano (ADA)', 'value': 'ADA'},
                {'label': 'WOZX', 'value': 'WOZX'},
                {'label': 'Stellar (XLM)', 'value': 'XLM'},
                {'label': 'Dogecoin (DOGE)', 'value': 'DOGE'},
                {'label': 'Polkadot (DOT)', 'value': 'DOT'},
                # 'YFI',
                
                
                                
            ],
            optionHeight=35,                    #height/space between dropdown options
            value='BTC',                    #dropdown value selected automatically when page loads
            disabled=False,                     #disable dropdown value selection
            multi=False,                        #allow multiple dropdown values to be selected
            searchable=True,                    #allow user-searching of dropdown values
            search_value='',                    #remembers the value searched in dropdown
            placeholder='Please select...',     #gray, default text shown when no option is selected
            clearable=True,                     #allow user to removes the selected value
            style={'width':"100%"},             #use dictionary to define CSS styles of your dropdown
            ),                                  #'memory': browser tab is refreshed
                                                #'session': browser tab is closed
                                                #'local': browser cookies are deleted

            html.Div(id='cryprocurrency_name_usd', style={'textAlign': 'center', 'color': "black", 'font-size' : "40px", }),
    # ],className='three columns'),

   
    # dcc.Tabs(id="tabs", children=[
        
        
    #     # Third Tab from here
    #     dcc.Tab(label='BitCoin Actual Data',children=[
            
        html.Div([
                dcc.Graph(id='our_graph', )
               ],className='eight columns',),

        html.Div([
        html.Div(className='three columns div-user-controls',
                             children=[
                             html.Br(),
                             html.Br(),
                             html.Br(),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                          # Data Table
                                         dash_table.DataTable(
                                             id='table',
                                             columns=[{"name": i, "id": i} for i in sorted(pred_table_data.columns)],
                                              style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                                                style_cell={
                                                    'backgroundColor': 'rgb(50, 50, 50)',
                                                    'color': 'white'
                                                    },
                                         ),
                                     ]),
                                ]
                             ),
         ],className='three columns'),


        


        # ]), # 3rd tab end ....!
            
        # ]),
        html.Br(),
        # html.H2("         Some Cryptocurrency Bar Graph Under 5000       "),
        html.Div([
            html.H2("         Some Cryptocurrency Bar Graph Under 5000 USD           "),
            ### Bar Chart
            dcc.Graph(id='bar_graph',
                figure = fig_bar,
                ),
            ], className='12 columns'),

        
    ])



#---------------------------------------------------------------
# Connecting the Dropdown values to the graph
@app.callback(
    Output('table', 'data'),
    Output('cryprocurrency_name_usd', 'children'),
    Output(component_id='our_graph', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)

def build_graph(column_chosen):
    
    # 0th index in Data dictionary is BTC
    if column_chosen == 'BTC':
        dfff = Last_DataFrame(BTC, btc_pred_data, btc_pred_index)
        pred_table_data = pd.DataFrame({
            "Date": btc_pred_index[-10:].values,
            "Prediction":btc_pred_data[:10]
            })

        last_day_price = BTC.tail(1)['close'].values[0]

        
        
        

    elif column_chosen == 'EOS':
        dfff = Last_DataFrame(EOS, eos_pred_data, eos_pred_index)
        pred_table_data = pd.DataFrame({
            "Date": eos_pred_index[-10:].values,
            "Prediction":eos_pred_data[:10]
            })
        last_day_price = EOS.tail(1)['close'].values[0]
        
        
    elif column_chosen == "ADA":
        dfff = Last_DataFrame(ADA, ada_pred_data, ada_pred_index)
        pred_table_data = pd.DataFrame({
            "Date": ada_pred_index[-10:].values,
            "Prediction":ada_pred_data[:10]
            })
        last_day_price = ADA.tail(1)['close'].values[0]
        
    elif column_chosen == "ETH":
        dfff = Last_DataFrame(ETH, eth_pred_data, eth_pred_index)
        pred_table_data = pd.DataFrame({
            "Date": eth_pred_index[-10:].values,
            "Prediction":eth_pred_data[:10]
            })
        last_day_price = ETH.tail(1)['close'].values[0]
    
    elif column_chosen == "LTC":
        dfff = Last_DataFrame(LTC, ltc_pred_data, ltc_pred_index)
        pred_table_data = pd.DataFrame({"Date": ltc_pred_index[-10:].values, "Prediction":ltc_pred_data[:10] })
        last_day_price = LTC.tail(1)['close'].values[0]
        
        
    elif column_chosen == "XRP":
        dfff = Last_DataFrame(XRP, xrp_pred_data, xrp_pred_index)
        pred_table_data = pd.DataFrame({"Date": xrp_pred_index[-10:].values, "Prediction":xrp_pred_data[:10] })
        last_day_price = XRP.tail(1)['close'].values[0]
        
    elif column_chosen == "BCH":
        dfff = Last_DataFrame(BCH, bch_pred_data, bch_pred_index)
        pred_table_data = pd.DataFrame({"Date": bch_pred_index[-10:].values, "Prediction":bch_pred_data[:10] })
        last_day_price = BCH.tail(1)['close'].values[0]
    
    elif column_chosen == "WOZX":
        dfff = Last_DataFrame(WOZX, wozx_pred_data, wozx_pred_index)
        pred_table_data = pd.DataFrame({"Date": wozx_pred_index[-10:].values, "Prediction":wozx_pred_data[:10] })
        last_day_price = WOZX.tail(1)['close'].values[0]
        
    elif column_chosen == "XLM":
        dfff = Last_DataFrame(XLM, xlm_pred_data, xlm_pred_index)
        pred_table_data = pd.DataFrame({"Date": xlm_pred_index[-10:].values, "Prediction":xlm_pred_data[:10] })
        last_day_price = XLM.tail(1)['close'].values[0]
        
    elif column_chosen == "DOGE":
        dfff = Last_DataFrame(DOGE, doge_pred_data, doge_pred_index)
        pred_table_data = pd.DataFrame({"Date": doge_pred_index[-10:].values, "Prediction":doge_pred_data[:10] })
        last_day_price = DOGE.tail(1)['close'].values[0]
    
    elif column_chosen == "DOT":
        dfff = Last_DataFrame(DOT, dot_pred_data, dot_pred_index)
        pred_table_data = pd.DataFrame({"Date": dot_pred_index[-10:].values, "Prediction":dot_pred_data[:10] })
        last_day_price = DOT.tail(1)['close'].values[0]
        
        
    fig = px.line(data_frame=dfff,
                      x="Date",
                      y=["Actual","Prediction"],
                      title='Actual Closing Price',
#                       mode='lines+markers'
                     )
    # fig.update_yaxes(ticklabelposition="inside left")
    fig.add_bar(x = dfff['Date'], y=dfff['Actual'], name='Actual Bar')

    # fig.layout.plot_bgcolor = 'rgb(50, 50, 50)'
    # fig.layout.paper_bgcolor = 'rgb(50, 50, 50)'
    # fig.layout.font = {'color': colors['text']}

    ccname_price = str(column_chosen) + " " + str(last_day_price) + " usd"
    

    return pred_table_data.to_dict('records'), ccname_price, fig


# In[ ]:


if __name__=='__main__':
    app.run_server(debug=True, port="8090")


# In[ ]:




