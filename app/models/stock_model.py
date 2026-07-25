from repository import StockRepository, ProcessedDataRepository, OrderbookRepository
from config import config
from flask import current_app as app
from database import StockDB
import pandas as pd

class StockModel:
    def __init__(self):
        self.repository = StockRepository()
        self.orderbook_repository = None
        self.processed_data_repository = None
            
        self.data = None

    def get_stocks(self, page, limit):
        return self.repository.get_stocks(page, limit)
    
    def get_ihsg_data(self, range_days):
        ihsg = self.repository.ihsg

        if ihsg is None:
            return []
        
        if ihsg.empty:
            return []
        
        if not pd.api.types.is_datetime64_any_dtype(ihsg['Date']):
            ihsg['Date'] = pd.to_datetime(ihsg['Date'])
        
        last_date = ihsg['Date'].max()
        start_date = last_date - pd.Timedelta(days=range_days) 
        filtered_ihsg = ihsg[ihsg['Date'] >= start_date]
        filtered_ihsg = filtered_ihsg.rename(columns={'Date': 'date'})
        filtered_ihsg['date'] = filtered_ihsg['date'].dt.strftime("%Y-%m-%d")

        result = []
        for _, filtered in filtered_ihsg.iterrows():
            result.append(filtered.to_dict())

        return result

    def get_stock_data(self, stock_name, range_days):
        if stock_name.lower() == "ihsg":
            result = self.get_ihsg_data(range_days)
            data = {
                "stock_data": result
            }
            return data

        if self.data is None:
            return []
        
        if self.data[stock_name.upper()] is None:
            return []
        
        data = self.data[stock_name.upper()]
        
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
        
        last_date = data['date'].max()
        start_date = last_date - pd.Timedelta(days=range_days) 
        filtered_data = data[data['date'] >= start_date]
        filtered_data['date'] = filtered_data['date'].dt.strftime("%Y-%m-%d")
        
        filtered_data = filtered_data[["date", "Close", "High", "Low", "Open", "Volume", "value", "offer_value", "bid_value", "bid_volume", "offer_volume", "foreign_sell", "foreign_buy", "sma5", "sma20", "sma50", "sma100", "wma5", "wma20", "wma50", "wma100", "ema5", "ema20", "ema50", "ema100", "lb_bolinger", "mb_bolinger", "ub_bolinger", "percent_bolinger", "return", "return_num", "macd_line", "macd_signal", "macd_hist", "rsi", "stoch_rsi_k%", "stoch_rsi_d%", "psar"]]

        result_dict = []
        for _, row in filtered_data.iterrows():
            result_dict.append(row.to_dict())

        data = {
            "stock_data": result_dict,
        }
        
        return data
        
    def processed_all_data(self):
        self.orderbook_repository = OrderbookRepository(is_init=True)
        self.processed_data_repository = ProcessedDataRepository(is_init=True)

        self.repository.load_ihsg()
        self.repository.load_ohlcv()
        
        self.orderbook_repository.load_orderbook_data()
        self.processed_data_repository.load_embedded_data()

        if self.processed_data_repository.processed_news is not None and self.orderbook_repository.orderbook is not None :
            stock_list = self.repository.get_all_stocks()
            with app.app_context():
                stocks = [
                    stock.code for stock in stock_list
                ]

            if len(stocks) > 0:
                self.data = {}
                
                for stock in stocks:
                    self.__load_data(stock)

    def __load_data(self, stock_name):
        if self.processed_data_repository.processed_news.get(stock_name) is not None and self.orderbook_repository.orderbook.get(stock_name) is not None and self.repository.ihsg is not None and self.repository.stock.get(stock_name) is not None:
            self.processed_data_repository.processed_news[stock_name]['date'] = pd.to_datetime(self.processed_data_repository.processed_news[stock_name]['date'], format="mixed").astype(config.astype_date_data)
            self.repository.stock[stock_name]['date'] = pd.to_datetime(self.repository.stock[stock_name]['date'], format="mixed").astype(config.astype_date_data)
            self.repository.ihsg['date'] = pd.to_datetime(self.repository.ihsg['date'], format="mixed").astype(config.astype_date_data)
            self.repository.ihsg['Close_ihsg'] = self.repository.ihsg['Close']
            self.orderbook_repository.orderbook[stock_name]['date'] = pd.to_datetime(self.orderbook_repository.orderbook[stock_name]['date'], format="mixed").astype(config.astype_date_data)

            self.data[stock_name] = pd.merge(self.repository.stock[stock_name], self.processed_data_repository.processed_news[stock_name], on='date', how='left', validate="one_to_one")
            self.data[stock_name] = pd.merge(self.data[stock_name], self.repository.ihsg[['date', 'Close_ihsg']], on='date', how='left', validate="one_to_one")
            self.data[stock_name] = pd.merge(self.data[stock_name], self.orderbook_repository.orderbook[stock_name][['date', 'value', 'offer_value', 'offer_volume', 'bid_value', 'bid_volume', 'foreign_sell', 'foreign_buy']], on='date', how='left', validate="one_to_one")
            
            numeric_cols = self.data[stock_name].select_dtypes('number').columns
            self.data[stock_name][numeric_cols] = self.data[stock_name][numeric_cols].fillna(0)
        else:
            self.data[stock_name] = None
    
    def get_all_stock_data(self, stock_name):
        if self.data is None:
            return []
        
        if self.data[stock_name.upper()] is None:
            return []
        
        data = self.data[stock_name.upper()]

        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
            
        data['date'] = data['date'].dt.strftime("%Y-%m-%d")

        result = []
        for _, filtered in data.iterrows():
            result.append(filtered.to_dict())

        return result

    def load_routine_stock(self):
        self.repository.save_ohlcv()
        self.repository.load_ohlcv()

        self.repository.save_ihsg()
        self.repository.load_ihsg()