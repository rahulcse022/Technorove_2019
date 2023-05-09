import pandas_datareader as pdr
import datetime
btc_data = pdr.get_data_yahoo(['BTC-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())['Close']
ada_data = pdr.get_data_yahoo(['ADA-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())['Close']
btc_data = pdr.get_data_yahoo(['BTC-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
ada_data = pdr.get_data_yahoo(['ADA-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
eth_data = pdr.get_data_yahoo(['ETH-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
xrp_data = pdr.get_data_yahoo(['XRP-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
xlm_data = pdr.get_data_yahoo(['XLM-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
# dot_data = pdr.get_data_yahoo(['DOT-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
neo_data = pdr.get_data_yahoo(['NEO-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
cel_data = pdr.get_data_yahoo(['CEL-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
bnb_data = pdr.get_data_yahoo(['BNB-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
ltc_data = pdr.get_data_yahoo(['LTC-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
xmr_data = pdr.get_data_yahoo(['XMR-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
# dai_data = pdr.get_data_yahoo(['DAI-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
vet_data = pdr.get_data_yahoo(['VET-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
eos_data = pdr.get_data_yahoo(['EOS-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
bsv_data = pdr.get_data_yahoo(['BSV-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
bch_data = pdr.get_data_yahoo(['BCH-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
btg_data = pdr.get_data_yahoo(['BTG-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
zec_data = pdr.get_data_yahoo(['ZEC-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
trx_data = pdr.get_data_yahoo(['TRX-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
xem_data = pdr.get_data_yahoo(['XEM-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
wozx_data = pdr.get_data_yahoo(['WOZX-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
wozx_data = pdr.get_data_yahoo(['WOZX-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
usdt_data = pdr.get_data_yahoo(['USDT-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
link_data = pdr.get_data_yahoo(['LINK-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
doge_data = pdr.get_data_yahoo(['DOGE-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())
dash_data = pdr.get_data_yahoo(['DASH-USD'],start=datetime.datetime(2020, 1, 1), end=datetime.date.today())


print(ada_data)