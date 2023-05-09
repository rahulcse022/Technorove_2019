import websocket
import json
import numpy as np

cc = 'btcusd'
interval = '1m'
socket = f'wss://stream.binance.com:9443/ws/{cc}t@kline_{interval}'

# Trading Strategy Parameters
aroon_time_period = 14
amount = 1000
core_trade_amount = amount*0.80
core_quantity = 0
trade_amount = amount*0.20
core_to_trade = True
portfolio = 0
investment, real_time_portfolio_value, closes, highs, lows = [], [], [], [], []
money_end = amount


# Buying and Selling functions

def buy(allocated_money, price):
  global portfolio, money_end
  quantity = allocated_money/price
  money_end -= quantity*price
  portfolio += quantity
  if investment == []:
    investment.append(allocated_money)
  else:
    investment.append(allocated_money)
    investment[-1] += investment[-2]

def sell(allocated_money, price):
  global money_end, portfolio
  quantity = allocated_money / price
  money_end += allocated_money
  portfolio -= quantity
  investment.append(-allocated_money)
  investment[-1] += investment[-2]


  # Bitcoin Bot

def on_close(ws):
  port_value = portfolio*closes[-1]
  if port_value > 0:
    sell(port_value,price = closes[-1])
  else:
    buy(-port_value, price = closes[-1])
  money_end += investment[-1]
  print('All trades settled')

def on_message(ws,message):
  global portfolio, investment, closes, highs, lows, money_end, core_to_trade, core_quantity, real_time_portfolio_value
  json_message = json.loads(message)
  cs = json_message['k']
  candle_closed, close, high, low = cs['x'], cs['c'], cs['h'], cs['l']

  if candle_closed:
    closes.append(float(close))
    highs.append(float(high))
    lows.append(float(low))
    last_price = closes[-1]
    print(f'Closes: {closes}')
    
    if core_to_trade:
      buy(core_trade_amount, price=closes[-1])
      print(f'Core Investment: We bought ${core_trade_amount} worth of bitcoin', '\n')
      core_quantity += core_trade_amount/closes[-1]
      core_to_trade = False

    aroon = talib.AROONOSC(np.array(highs), np.array(lows), aroon_time_period)
    last_aroon = round(aroon[-1],2)
    amt = last_aroon*trade_amount/100
    port_value = portfolio*last_price - core_quantity*last_price
    trade_amt = amt - port_value
    RT_portfolio_value = money_end + port_value + core_quantity*last_price
    real_time_portfolio_value.append(float(RT_portfolio_value))
    print(f'The Last Aroon is "{last_aroon}" and recommended exposure is "${amt}"')
    print(f'Real-Time Portfolio Value: ${RT_portfolio_value}', '\n')
    
    if trade_amt > 0:
      buy(trade_amt, price=last_price)
      print(f'We bought ${trade_amt} worth of bitcoin', '\n', '\n')
    elif trade_amt < 0:
      sell(-trade_amt, price=last_price)
      print(f'We sold ${-trade_amt} worth of bitcoin', '\n', '\n')

ws = websocket.WebSocketApp(socket, on_message=on_message, on_close=on_close)
