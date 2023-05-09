import websocket
import json
import numpy as np
import csv
from datetime import datetime


print("\nProgram started >>>> \n")
# print(datetime.fromtimestamp(int(16207164600)))
cc = 'btcusd'
interval = '1m'

socket = f'wss://stream.binance.com:9443/ws/{cc}t@kline_{interval}'


def on_close(ws):
	print("on_close")


def on_message(ws,message):

  json_message = json.loads(message)
  cs = json_message['k']
  candle_closed, close, T  = cs['x'], cs['c'], cs['t']

  if candle_closed:
  	print("candle_closed started")
  	print(T)
  	T = T//100
  	print(T)
  	print(datetime.fromtimestamp(int(T)))
  	d = str(datetime.datetime.fromtimestamp(T))
  	with open("rahul.csv", 'w', newline='') as file:
  		writer = csv.writer(file)
  		writer.writerow(["SN", "Transaction"])
  		writer.writerow([1, d])


ws = websocket.WebSocketApp(socket, on_message=on_message, on_close=on_close)
ws.run_forever()