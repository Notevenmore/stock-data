import yfinance as yf
import pandas as pd
import numpy as np
from config import config
from database import StockDB
from flask import current_app as app

class StockRepository:
    def __init__(self, is_init = False):
        self.ihsg = None
        self.stock = {}

        if is_init:
            self.load_ohlcv()
            self.load_ihsg()
    
    def get_all_stocks(self):
        with app.app_context():
            stocks = StockDB.query.order_by(StockDB.id).all()
        
        return stocks

    def get_stocks(self, page, limit):
        offset = (page-1) * limit
        with app.app_context():
            total = StockDB.query.count()
            stocks = [stock.to_dict() for stock in StockDB.query.order_by(StockDB.id).offset(offset).limit(limit).all()]
        
        for stock in stocks:
            stock_df = self.stock.get(stock['code'])
            if stock_df is None:
                stock["current_price"] = 0
                continue
            
            if stock_df.empty:
                stock["current_price"] = 0
                continue

            stock_data = stock_df.iloc[-1]
            stock["current_price"] = stock_data["Close"]
        

        data = {
            "stocks": stocks,
            "page": page,
            "limit": limit,
            "total": total,
            "has_next": offset + limit < total
        }

        return data

    def save_ohlcv(self):
        with app.app_context():
            stocks = [
                stock.code for stock in StockDB.query.all()
            ]

        for stock in stocks:
            df = yf.download(stock+'.JK', start=config.start, end=config.end)
            df.columns = df.columns.droplevel(1)
            df.columns.name = None
            df = df.reset_index()
            df.to_csv(f"data/Technical/{stock}.csv")

    def save_ihsg(self):
        df_ihsg = yf.download('^JKSE', start=config.start, end=config.end)
        df_ihsg.columns = df_ihsg.columns.droplevel(1)
        df_ihsg.columns.name = None
        df_ihsg = df_ihsg.reset_index()
        df_ihsg.to_csv("data/IHSG.csv")

    def load_ohlcv(self):
        with app.app_context():
            stocks = [
                stock.code for stock in StockDB.query.all()
            ]

        for stock in stocks:
            try:
                self.stock[stock] = pd.read_csv(f'data/Technical/{stock}.csv')
                self.stock[stock].drop(columns=['Unnamed: 0'], inplace=True)
                self.stock[stock].rename(columns={'Date': 'date'}, inplace=True)
                self.__generate_technical_data(stock)
            except FileNotFoundError:
                self.stock[stock] = None

    def load_ihsg(self):
        try:
            self.ihsg = pd.read_csv('data/IHSG.csv')
            self.ihsg.drop(columns=['Unnamed: 0'], inplace=True)
            self.ihsg.rename(columns={'Date': 'date'}, inplace=True)
        except FileNotFoundError:
            self.ihsg = None

    def __generate_technical_data(self, stock_name):
        if self.stock is not None:
            if self.stock[stock_name] is not None:
                if len(self.stock[stock_name]) > 0:
                    self.stock[stock_name]['sma5'], self.stock[stock_name]['wma5'], self.stock[stock_name]['ema5'] = self.__compute_ma(self.stock[stock_name]['Close'], 5)
                    self.stock[stock_name]['sma20'], self.stock[stock_name]['wma20'], self.stock[stock_name]['ema20'] = self.__compute_ma(self.stock[stock_name]['Close'], 20)
                    self.stock[stock_name]['sma50'], self.stock[stock_name]['wma50'], self.stock[stock_name]['ema50'] = self.__compute_ma(self.stock[stock_name]['Close'], 50)
                    self.stock[stock_name]['sma100'], self.stock[stock_name]['wma100'], self.stock[stock_name]['ema100'] = self.__compute_ma(self.stock[stock_name]['Close'], 100)

                    # Bolinger band
                    self.stock[stock_name]['lb_bolinger'], self.stock[stock_name]['mb_bolinger'], self.stock[stock_name]['ub_bolinger'], self.stock[stock_name]['percent_bolinger'] = self.__compute_bolinger_band(self.stock[stock_name]['Close'])

                    # return harian
                    self.stock[stock_name]['return'] = self.__compute_return(self.stock[stock_name]['Close'])
                    self.stock[stock_name]['return_num'] = self.__compute_difference(self.stock[stock_name]['Close'])

                    # MACD
                    self.stock[stock_name]['macd_line'], self.stock[stock_name]['macd_signal'], self.stock[stock_name]['macd_hist'] = self.__compute_macd(self.stock[stock_name]['Close'])

                    # RSI & Stoch RSI
                    self.stock[stock_name]['rsi'] = self.__compute_rsi(self.stock[stock_name]['Close'])
                    self.stock[stock_name]['stoch_rsi_k%'], self.stock[stock_name]['stoch_rsi_d%'] = self.__compute_stoch_rsi(self.stock[stock_name]['rsi'])

                    # Parabolic Stop and Reversal
                    self.stock[stock_name]['psar'] = self.__compute_psar(self.stock[stock_name])

                    # df['return_ihsg'] = compute_return(df['Close_ihsg'])
    
    def __compute_rsi(self, series, period = 14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (100 + rs))
        return rsi

    def __compute_stoch_rsi(self, rsi, lookback = 14):
        min_rsi = rsi.rolling(window=lookback).min()
        max_rsi = rsi.rolling(window=lookback).max()

        range_rsi = max_rsi - min_rsi
        k = ((rsi - min_rsi) / range_rsi) * 100
        d = k.rolling(window = 3).mean()
        
        return k, d

    def __compute_macd(self, series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_hist

    def __compute_ma(self, series, n):
        sma = series.rolling(window = n).mean()
        
        wma = series.rolling(window = n).apply(lambda window: sum(window[i] * (n - (i + 1)) for i in range(n)), raw=True) 
        wma = wma / ((n * (n + 1))/2)

        alpha  = 2 / (n + 1)
        ema = series.copy().astype(float)
        ema.iloc[0] = series.iloc[0]
        for i in range(1, len(series)):
            ema.iloc[i] = series.iloc[i] * alpha + ema.iloc[i-1] * (1 - alpha)
        
        return sma, wma, ema

    def __compute_bolinger_band(self, series, n = 20, k = 2):
        mb = series.rolling(window=n).mean()
        volatile = np.sqrt((((series - mb) ** 2)).rolling(window = n).mean())
        lb = mb - (k * volatile)
        ub = mb + (k * volatile)
        pbb = (series - lb) / (ub - lb)

        return lb, mb, ub, pbb

    def __compute_return(self, series, t = 1):
        result = [series.iloc[i] / series.iloc[i - t] for i in range(len(series))]
        result = np.log(result) * 100
        return result

    def __compute_difference(self, series, t = 1):
        result = series - series.shift(t)
        return result

    def __compute_psar(self, data, af_start = 0.02, af_step = 0.02, af_max = 0.2):
        psar  = pd.Series(index=data['High'].index, dtype=float)
        trend = 1
        af = af_start
        ep = None
        psar.iloc[1] = np.nan

        if data['High'].iloc[1] > data['High'].iloc[0]:
            trend = 1
            ep = data['High'].iloc[1]
            psar.iloc[1] = data['Low'].iloc[0]
        else:
            trend = -1
            ep = data['Low'].iloc[1]
            psar.iloc[1] = data['High'].iloc[0]
        
        for i in range(2, len(data['High'])):
            sar_prev = psar.iloc[i-1]
            if trend == 1:
                psar.iloc[i] = sar_prev + (af * (ep - sar_prev))
                psar.iloc[i] = min(psar.iloc[i], data['Low'].iloc[i-1], data['Low'].iloc[i-2] if i>1 else data['Low'].iloc[i-1])
                if psar.iloc[i] > data['Low'].iloc[i]:
                    trend = -1
                    psar.iloc[i] = ep
                    af = af_start
                    ep = data['Low'].iloc[i]
                    sar_prev = psar.iloc[i]
                    if trend == -1:
                        psar.iloc[i] = sar_prev + (af * (ep - sar_prev))
                        psar.iloc[i] = max(psar.iloc[i], data['High'].iloc[i - 1], data['High'].iloc[i - 2] if i > 1 else data['High'].iloc[i - 1])
                        continue
            else:
                psar.iloc[i] = sar_prev + (af * (ep - sar_prev))
                psar.iloc[i] = max(psar.iloc[i], data['High'].iloc[i-1], data['High'].iloc[i-2] if i > 1 else data['High'].iloc[i-1])
                if psar.iloc[i] < data['High'].iloc[i]:
                    trend = 1
                    psar.iloc[i] = ep
                    af = af_start
                    ep = data['High'].iloc[i]
                    sar_prev = psar.iloc[i]
                    if trend == 1:
                        psar.iloc[i] = sar_prev + (af * (ep - sar_prev))
                        psar.iloc[i] = min(psar.iloc[i], data['Low'].iloc[i-1], data['Low'].iloc[i-2] if i < 1 else data['Low'].iloc[i-1])
                        continue
                
            if trend == 1:
                if data['High'].iloc[i] > ep:
                    ep = data['High'].iloc[i]
                    af = min(af + af_step, af_max)
            else:
                if data['Low'].iloc[i] < ep:
                    ep = data['Low'].iloc[i]
                    af = min(af + af_step, af_max)
        
        return psar

