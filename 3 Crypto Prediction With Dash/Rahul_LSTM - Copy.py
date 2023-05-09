import requests
from datetime import datetime
import pandas as pd
import numpy as np
import json
import tensorflow as tf
### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

import pickle







def Data_Fetching(currency_symbol, to_currency, limit):
	""" Example : Currency_symbole : "BTC"
				  to_currency : "USD"
                  limit  : 1000
	"""
	endpoint = 'https://min-api.cryptocompare.com/data/histoday'
	res = requests.get(endpoint + '?fsym={0}&tsym={1}&limit={2}'.format(currency_symbol, to_currency, limit))
	hist = pd.DataFrame(json.loads(res.content)['Data'])
	hist.drop(axis=1, columns=['volumefrom','volumeto','conversionType','conversionSymbol'], inplace=True)
	hist.dropna(axis=0,how="any",inplace=True)
	
	hist = hist.set_index('time')
	hist.index = pd.to_datetime(hist.index, unit='s')
	hist['datetime'] = hist.index
	col = {"high":"High", "low":"Low" , "open": "Open", "close":"Close" , "datetime":"DateTime" }
	hist.rename(columns=col, inplace=True)
	hist.index.names = ['Date']
	

	return hist
dfff = Data_Fetching("BTC", "USD", "1000")

# df['Close'].plot()




def Predication_Next_Days(df, split_data, tat_size, time_step, future_days):
	"""
	Example : 
				# dff = df['Close']
				# splite = 1000 max .....   how many last days you are considering
				# training_and_testing_size = 0.90  ### Rahul
				# time_step = n_steps =  5  # Rahul
				# future_days = 5

	"""

	df_d = df[-split_data:].copy()

	from sklearn.preprocessing import MinMaxScaler
	scaler=MinMaxScaler(feature_range=(0,1))
	df_d=scaler.fit_transform(np.array(df_d).reshape(-1,1))


	##splitting dataset into train and test split
	training_size=int(len(df_d)*tat_size)
	test_size=len(df_d)-training_size
	train_data,test_data=df_d[0:training_size,:],df_d[training_size:len(df_d),:1]


	import numpy
	# convert an array of values into a dataset matrix
	def create_dataset(dataset, time_step=1):
		dataX, dataY = [], []
		for i in range(len(dataset)-time_step-1):
			a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
			dataX.append(a)
			dataY.append(dataset[i + time_step, 0])
		return numpy.array(dataX), numpy.array(dataY)


	X_train, y_train = create_dataset(train_data, time_step)
	X_test, y_test = create_dataset(test_data, time_step)


	# reshape input to be [samples, time steps, features] which is required for LSTM
	X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
	X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


	model=Sequential()
	model.add(LSTM(50,return_sequences=True,input_shape=(X_test.shape[1],1)))
	model.add(LSTM(50,return_sequences=True))
	model.add(LSTM(50))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error',optimizer='adam')


	model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=64,verbose=1)


	### Lets Do the prediction
	train_predict=model.predict(X_train)
	test_predict=model.predict(X_test)


	##Transformback to original form
	train_predict=scaler.inverse_transform(train_predict)
	test_predict=scaler.inverse_transform(test_predict)



	x_input=test_data[len(test_data)-time_step:].reshape(1,-1)

	temp_input=list(x_input)
	temp_input=temp_input[0].tolist()

	# demonstrate prediction for next 10 days
	from numpy import array

	lst_output=[]
	i=0
	while(i<future_days):
	    
	    if(len(temp_input)>time_step):
	        x_input=np.array(temp_input[1:])
	        x_input=x_input.reshape(1,-1)
	        x_input = x_input.reshape((1, time_step, 1))
	        yhat = model.predict(x_input, verbose=0)
	        temp_input.extend(yhat[0].tolist())
	        temp_input=temp_input[1:]
	        lst_output.extend(yhat.tolist())
	        i=i+1
	    else:
	        x_input = x_input.reshape((1, time_step,1))
	        yhat = model.predict(x_input, verbose=0)
	        temp_input.extend(yhat[0].tolist())
	        lst_output.extend(yhat.tolist())
	        i=i+1
	# model.save("R_Model.h5")

	# Save the trained model as a pickle string.
	saved_model_btc = pickle.dumps(model)


	return scaler.inverse_transform(lst_output)


print(Predication_Next_Days(dfff['Close'], 1000, 0.70, 5, 5))


