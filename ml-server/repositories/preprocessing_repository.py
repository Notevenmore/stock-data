import requests
import pandas as pd
import numpy as np
import ast

class PreprocessingRepository:
    def __init__(self):
        self.url = "http://127.0.0.1:5000"
        self.stock_list = pd.DataFrame()
        self.x_train = {}
        self.y_train = {}
        self.x_test = {}
        self.y_test = {}
        self.scale_columns = [
            "Volume",
            "value",
            "bid_volume",
            "offer_volume",
            "bid_value",
            "offer_value",
            "foreign_sell",
            "foreign_buy",
            "foreign_net",
            "Open",
            "High",
            "Low",
            "Close",
            'Close_ihsg',
            'ema100', 'ema20', 'ema5', 'ema50',
            'sma100', 'sma20', 'sma5', 'sma50',
            'wma100', 'wma20', 'wma5', 'wma50',
            'lb_bolinger', 'mb_bolinger', 'ub_bolinger',
            'macd_hist', 'macd_line', 'macd_signal',
            'psar'
        ]

        self.load_list_stock()


    def load_list_stock(self):
        response = requests.get(f"{self.url}/stock")
        json = response.json()
        data = json.get('data')
        self.stock_list = pd.DataFrame(data)

    def init_data(self):
        for _, stock in self.stock_list.iterrows():
            self.load_company_data(stock['code'])

    def load_company_data(self, stock_name):
        response = requests.get(f'{self.url}/stock/ml/{stock_name}')
        json = response.json()
        data = json.get('data')

        df = pd.DataFrame(data)
        df = df.drop(['attention_mask', 'content', 'content_abbreviation', 'content_clean', 'content_entity', 'content_stemmed', 'content_stopword_removed', 'emb_mean', 'embedding', 'sentiment_label', 'full_text', 'input_ids', 'negative_grouped_news_sentiment', 'positive_grouped_news_sentiment', 'title', 'title_abbreviation', 'title_clean', 'title_entity', 'title_stemmed', 'title_stopword_removed'], axis=1)
        df["sentiment_score"] = df["sentiment_score"].apply(
            lambda x: [] if pd.isna(x) else ast.literal_eval(x)
        )
        df["foreign_net"] = df["foreign_buy"] - df["foreign_sell"]
        df["Volume"] = np.log1p(df["Volume"])
        df["value"] = np.log1p(df["value"])
        df["foreign_buy"] = np.log1p(df["foreign_buy"])
        df["foreign_sell"] = np.log1p(df["foreign_sell"])
        df["bid_volume"] = np.log1p(df["bid_volume"])
        df["offer_volume"] = np.log1p(df["offer_volume"])

        
        features = df.columns.drop(['date', 'return', 'return_num', 'sentiment_score', *[f"sentiment_pca_{i}" for i in range(50)]])
        x = df[features]
        y = df['return_num']

        split_index = int(len(df) * 0.8)
        x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        self.x_train[stock_name] = x_train
        self.x_test[stock_name] = x_test
        self.y_train[stock_name] = y_train
        self.y_test[stock_name] = y_test
