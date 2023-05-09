import websocket
import json
import numpy as np
import csv
from  datetime import datetime



print("\nProgram started >>>> \n")
cc = 'btcusd'
interval = '1m'

socket = f'wss://stream.binance.com:9443/ws/{cc}t@kline_{interval}'

investment, real_time_portfolio_value, closes = [], [], []
sell_pur_hist = []
portfolio = 0
a, b, c = [], [], []
stock_avg_price = 0
amount = 2000
core_trade_amount = amount*0.80
core_quantity = 0
trade_amount = amount*0.20
core_to_trade = True
indication_to_buy, indication_to_sell = False, False

portfolio = 0
investment, real_time_portfolio_value, closes, highs, lows = [], [], [], [], []
money_end = amount


##*********** Variables for Dataset  ***************** ##
t_time, T_time, Transaction, Closing_Price, Sum_of_a, Sum_of_c = [],[],[],[],[],[]
Avg, Investment, Portfolio, Money_end, RT_portfolio = [],[],[],[],[]
transaction = ''
prev_avg = 0
with open("BTC_Database91.csv", 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["SN", "Transaction", "Transaction Price", "Closing Price", "Average", "Investment", "Portfolio", "Money end", "Real Time Portfolio Vlaue"])





def buy(allocated_money, price):
  global portfolio, money_end ,investment, transaction, transaction_price
  transaction = "Buy"
  transaction_price = -allocated_money
  print(f"Buy ${allocated_money} at price {price}")
  quantity = allocated_money/price
  money_end -= quantity*price
  portfolio += quantity
  if investment == []:
    investment.append(allocated_money)
  else:
    investment.append(allocated_money)
    investment[-1] += investment[-2]

def sell(allocated_money, price):
  global money_end, portfolio ,investment, transaction, transaction_price
  transaction_price = allocated_money
  transaction = "Sell"
  print(f"Sell ${allocated_money} at price {price}")
  quantity = allocated_money / price
  money_end += allocated_money
  portfolio -= quantity
  investment.append(-allocated_money)
  investment[-1] += investment[-2]


# Bitcoin Bot
def on_close(ws):
	global a,b,c
	print("a,b,c :",a,b,c)
	print("Program Closed <<< ")

def on_message(ws,message):
  global portfolio, investment, closes, highs, lows, money_end, core_to_trade, core_quantity, real_time_portfolio_value
  global stock_avg_price, a, b, c, sell_pur_hist
  global t_time, prev_avg


  json_message = json.loads(message)
  cs = json_message['k']
  candle_closed, close, t, T  = cs['x'], cs['c'], cs['t'], cs['T']

  if candle_closed:
    closes.append(float(close))
    last_price = closes[-1]
    print(f'{len(closes)} Closes: {closes[-20:]}')

    if len(closes) == 1:
    	buy(100, closes[-1])
    	sell_pur_hist.append(["Buy", 100, closes[-1]])
    	stock_avg_price = closes[-1]
    	a.append(100/closes[-1]) # a = 5
    	b.append(closes[-1])    # b = 360
    	c.append(a[-1]*b[-1])    # c = 1800 = 5*360
    	stock_avg_price = sum(c)/sum(a)
    	print("a,b,c : ",a,b,c) ### a, b, c == 5stock at price 360 then total price 1800
    	print('stock avg price',stock_avg_price)
    else:
    	if stock_avg_price > closes[-1] or stock_avg_price == 0 :
            if money_end > 0 :
                indication_to_buy = True
                buy(100, closes[-1])
                # transaction = "Buy"
                sell_pur_hist.append(["Buy", 100, closes[-1]])
                a.append(100/closes[-1]) # number of stock buy
                b.append(closes[-1])
                c.append(a[-1]*b[-1])
                # if stock_avg_price == 0:
                # 	stock_avg_price = closes[-1]
                # else:
                stock_avg_price = sum(c)/sum(a)
                indication_to_buy = False
                indication_to_sell = False
            else:
              print("Insuficiant Balance")
    	else:# stock_avg_price < closes[-1] and stock_avg_price > 0: # Sell
    		# print(f"""
    		# 	stock_avg_price < close_price
    		# 	{stock_avg_price} < {closes[-1]}
    		# 	""")
    		indication_to_sell = True
    		sell(sum(a)*stock_avg_price, closes[-1])
	    	sell_pur_hist.append(["Sell", sum(a)*stock_avg_price, closes[-1]])
	    	a,b,c = [],[],[]
	    	# sell half of the current stock quantity
	    	# a.append(-sum(c)/closes[-1])
	    	# b.append(closes[-1])
	    	# c.append(a[-1]*b[-1])  # it will autometically minus cuase of -a
	    	stock_avg_price = 0 #sum(c)/sum(a)
	    	indication_to_sell = False
	    	indication_to_buy = False

    # print("*****************line-end*****************")
    # print("Sell Pur Hist", sell_pur_hist)
    # print("stock_avg_price : ", stock_avg_price)
    RT_portfolio_value = money_end + portfolio*closes[-1]
    print("Portfolio : ",portfolio)
    print("Money End : ",money_end)
    print("Real Time Portfolio Value : ",RT_portfolio_value)
    print("*****************line-end*****************\n\n")
    # T = str(datetime.datetime.fromtimestamp(T))

    with open("BTC_Database91.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([len(closes), transaction, transaction_price, closes[-1], prev_avg, investment[-1], portfolio, money_end, RT_portfolio_value]) # "Investment", "Portfolio", "Money end", "Real Time Portfolio Vlaue"
    prev_avg = stock_avg_price
    if RT_portfolio_value < amount-2:
        ws.close()

# writer.writerow(["Closing Price", "Sum_of_a", "Sum_of_c", "Average", "Investment", "Portfolio", "Money end", "Real Time Portfolio Vlaue"])
ws = websocket.WebSocketApp(socket, on_message=on_message, on_close=on_close)
ws.run_forever()