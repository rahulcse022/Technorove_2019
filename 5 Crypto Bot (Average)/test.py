import websocket
import json
import numpy as np
print("\nProgram started >>>> \n")
cc = 'btcusd'
interval = '1m'

socket = f'wss://stream.binance.com:9443/ws/{cc}t@kline_{interval}'

investment, real_time_portfolio_value, closes = [], [], []
sell_pur_hist = []
portfolio = 0
a, b, c = [], [], []
stock_avg_price = 0
amount = 1000
core_trade_amount = amount*0.80
core_quantity = 0
trade_amount = amount*0.20
core_to_trade = True
indication_to_buy, indication_to_sell = False, False

portfolio = 0
investment, real_time_portfolio_value, closes, highs, lows = [], [], [], [], []
money_end = amount



def buy(allocated_money, price):
  global portfolio, money_end ,investment
  print(f"Buy {allocated_money} at price {price}")
  quantity = allocated_money/price
  money_end -= quantity*price
  portfolio += quantity
  if investment == []:
    investment.append(allocated_money)
  else:
    investment.append(allocated_money)
    investment[-1] += investment[-2]

def sell(allocated_money, price):
  global money_end, portfolio ,investment
  print(f"Sell {allocated_money} at price {price}")
  quantity = allocated_money / price
  money_end += allocated_money
  portfolio -= quantity
  investment.append(-allocated_money)
  investment[-1] += investment[-2]


# Bitcoin Bot
def on_close(ws):
  print("On_closeing")

def on_message(ws,message):
  global portfolio, investment, closes, highs, lows, money_end, core_to_trade, core_quantity, real_time_portfolio_value
  global stock_avg_price, a, b, c, sell_pur_hist
  json_message = json.loads(message)
  cs = json_message['k']
  candle_closed, close = cs['x'], cs['c']

  if candle_closed:
    closes.append(float(close))
    last_price = closes[-1]
    print(f'{len(closes)} Closes: {closes}')

    if len(closes) == 1:
    	buy(100, closes[-1])
    	print("1st Buy $100 BTC at price :", closes[-1])
    	sell_pur_hist.append(["Buy", 100, closes[-1]])
    	print("sell pur hsit", sell_pur_hist)
    	stock_avg_price = closes[-1]
    	a.append(100/closes[-1]) # a = 5
    	b.append(closes[-1])    # b = 360
    	c.append(a[-1]*b[-1])    # c = 1800 = 5*360
    	print("a,b,c : ",a,b,c) ### a, b, c == 5stock at price 360 then total price 1800
    	print('stock avg price',stock_avg_price,'\n\n')
    else:
    	if stock_avg_price > closes[-1]:
    		indication_to_buy = True
    		print("buy indication on")
    	if stock_avg_price < closes[-1]:
    		indication_to_sell = True
    		print("sell indiation On")

    	if indication_to_buy:
    		print("we are in here to buy btc")
    		buy(100, closes[-1])
	    	print("Buy $100 BTC at price :", closes[-1])
	    	sell_pur_hist.append(["Buy", 100, closes[-1]])
	    	a.append(100/closes[-1])
	    	b.append(closes[-1])
	    	c.append(a[-1]*b[-1])
	    	stock_avg_price = sum(c)/sum(a)
	    	print("stock_avg_price indiation to buy", stock_avg_price)
	    	indication_to_buy = False
	    	indication_to_sell = False

    	if indication_to_sell:
    		print("we are in here to sell btc")
    		sell(sum(c)/2, closes[-1])  # sell half of the tottal stock
	    	print("Sell $100 BTC at price :",closes[-1])
	    	sell_pur_hist.append(["Sell", 100, closes[-1]])
	    	# sell half of the current stock quantity
	    	a.append(-(sum(c)/2))
	    	b.append(closes[-1])
	    	c.append(a[-1]*b[-1])  # it will autometically minus cuase of -a
	    	stock_avg_price = sum(c)/sum(a)
	    	indication_to_sell = False
	    	indication_to_buy = False
	    	


    print("*****************line-end*****************")
    print("Sell Pur Hist", sell_pur_hist)
    print("*****************line-end*****************")
    
    
    # print(a,b,c, stock_avg_price)
    # print(sell_pur_hist)

    
ws = websocket.WebSocketApp(socket, on_message=on_message, on_close=on_close)
ws.run_forever()