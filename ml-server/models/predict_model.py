from repositories import GRURepository, LSTMRepository, NANNRepository, RFRRepository, PreprocessingRepository

class PredictModel:
    def __init__(self):
        self.preprocessing_repository = None
        self.gru_repository = None
        self.lstm_repository = None
        self.nann_repository = None
        self.rfr_repository = None

        self.gru_metric_score = {}
        self.lstm_metric_score = {}
        self.nann_metric_score = {}
        self.rfr_metric_score = {}

        self.best_model_predicted = {}
        self.best_metadatas = {}
        self.next_price = {}

        self.r2_weight = 0.4
        self.accuracy_weight = 0.3
        self.mae_weight = 0.2
        self.mse_weight = 0.1

    def init_preprocessing_repository(self):
        self.preprocessing_repository = PreprocessingRepository()
        self.preprocessing_repository.init_data()

    def init_gru_repository(self):
        self.gru_repository = GRURepository(self.preprocessing_repository)

    def init_lstm_repository(self):
        self.lstm_repository = LSTMRepository(self.preprocessing_repository)

    def init_nann_repository(self):
        self.nann_repository = NANNRepository(self.preprocessing_repository)

    def init_rfr_repository(self):
        self.rfr_repository = RFRRepository(self.preprocessing_repository)

    def analyze_all(self):
        self.init_preprocessing_repository()
        self.init_gru_repository()
        self.init_lstm_repository()
        self.init_nann_repository()
        self.init_rfr_repository()

        self.gru_repository.workflow_analyze_stocks()
        self.lstm_repository.workflow_analyze_stocks()
        self.rfr_repository.workflow_analyze_stocks()
        self.nann_repository.workflow_analyze_stocks()

        self.load_best_predicted_datas()

    def load_best_predicted_datas(self):
        self.init_preprocessing_repository()
        self.gru_repository.load_metadatas()
        self.lstm_repository.load_metadatas()
        self.rfr_repository.load_metadatas()
        self.nann_repository.load_metadatas()

        list_stock = self.preprocessing_repository.stock_list
        if not list_stock.empty and self.gru_repository.metadata and self.lstm_repository.metadata and self.rfr_repository.metadata and self.nann_repository.metadata:
            for _, stock in list_stock.iterrows():
                self.__compare_models(stock['code'])

    def __compare_models(self, stock_name):
        self.gru_metric_score[stock_name] = {}
        self.lstm_metric_score[stock_name] = {}
        self.rfr_metric_score[stock_name] = {}
        self.nann_metric_score[stock_name] = {}

        gru_metadata = self.gru_repository.metadata[stock_name]
        lstm_metadata = self.lstm_repository.metadata[stock_name]
        rfr_metadata = self.rfr_repository.metadata[stock_name]
        nann_metadata = self.nann_repository.metadata[stock_name]

        self.__normalization_metrics(
            stock_name,
            gru_metadata,
            lstm_metadata,
            rfr_metadata,
            nann_metadata
        )

        self.__count_weighted_score(stock_name)
        self.__ranking_model_and_get_next_price(stock_name)

    def __normalization_metric_r2(self, stock_name, gru_r2, lstm_r2, rfr_r2, nann_r2):
        r2_scores = []
        if gru_r2 != None:
            r2_scores.append(gru_r2)
        if lstm_r2 != None:
            r2_scores.append(lstm_r2)
        if rfr_r2 != None:
            r2_scores.append(rfr_r2)
        if nann_r2 != None:
            r2_scores.append(nann_r2)

        max_r2 = max(r2_scores)
        min_r2 = min(r2_scores)

        self.gru_metric_score[stock_name]["r2"] = (gru_r2 - min_r2) / (max_r2 - min_r2)
        self.lstm_metric_score[stock_name]["r2"] = (lstm_r2 - min_r2) / (max_r2 - min_r2)
        self.rfr_metric_score[stock_name]["r2"] = (rfr_r2 - min_r2) / (max_r2 - min_r2)
        self.nann_metric_score[stock_name]["r2"] = (nann_r2 - min_r2) / (max_r2 - min_r2)


    def __normalization_metric_direction_accuracy(self, stock_name, gru_accuracy, lstm_accuracy, rfr_accuracy, nann_accuracy):
        accuracy_scores = []
        if gru_accuracy != None:
            accuracy_scores.append(gru_accuracy)
        if lstm_accuracy != None:
            accuracy_scores.append(lstm_accuracy)
        if rfr_accuracy != None:
            accuracy_scores.append(rfr_accuracy)
        if nann_accuracy != None:
            accuracy_scores.append(nann_accuracy)

        max_accuracy = max(accuracy_scores)
        min_accuracy = min(accuracy_scores)

        self.gru_metric_score[stock_name]["accuracy"] = (gru_accuracy - min_accuracy) / (max_accuracy - min_accuracy)
        self.lstm_metric_score[stock_name]["accuracy"] = (lstm_accuracy - min_accuracy) / (max_accuracy - min_accuracy)
        self.rfr_metric_score[stock_name]["accuracy"] = (rfr_accuracy - min_accuracy) / (max_accuracy - min_accuracy)
        self.nann_metric_score[stock_name]["accuracy"] = (nann_accuracy - min_accuracy) / (max_accuracy - min_accuracy)

    def __normalization_metric_mse(self, stock_name, gru_mse, lstm_mse, rfr_mse, nann_mse):
        mse_scores = []
        if gru_mse != None:
            mse_scores.append(gru_mse)
        if lstm_mse != None:
            mse_scores.append(lstm_mse)
        if rfr_mse != None:
            mse_scores.append(rfr_mse)
        if nann_mse != None:
            mse_scores.append(nann_mse)

        max_mse = max(mse_scores)
        min_mse = min(mse_scores)

        self.gru_metric_score[stock_name]["mse"] = (max_mse - gru_mse) / (max_mse - min_mse)
        self.lstm_metric_score[stock_name]["mse"] = (max_mse - lstm_mse) / (max_mse - min_mse)
        self.rfr_metric_score[stock_name]["mse"] = (max_mse - rfr_mse) / (max_mse - min_mse)
        self.nann_metric_score[stock_name]["mse"] = (max_mse - nann_mse) / (max_mse - min_mse)

    def __normalization_metric_mae(self, stock_name, gru_mae, lstm_mae, rfr_mae, nann_mae):
        mae_scores = []
        if gru_mae != None:
            mae_scores.append(gru_mae)
        if lstm_mae != None:
            mae_scores.append(lstm_mae)
        if rfr_mae != None:
            mae_scores.append(rfr_mae)
        if nann_mae != None:
            mae_scores.append(nann_mae)

        max_mae = max(mae_scores)
        min_mae = min(mae_scores)

        self.gru_metric_score[stock_name]["mae"] = (max_mae - gru_mae) / (max_mae - min_mae)
        self.lstm_metric_score[stock_name]["mae"] = (max_mae - lstm_mae) / (max_mae - min_mae)
        self.rfr_metric_score[stock_name]["mae"] = (max_mae - rfr_mae) / (max_mae - min_mae)
        self.nann_metric_score[stock_name]["mae"] = (max_mae - nann_mae) / (max_mae - min_mae)

    def __normalization_metrics(self, stock_name, gru_metadata, lstm_metadata, rfr_metadata, nann_metadata):
        self.__normalization_metric_r2(
            stock_name, 
            gru_metadata["metrics"]["latest"]["r2"], 
            lstm_metadata["metrics"]["latest"]["r2"], 
            rfr_metadata["metrics"]["latest"]["r2"], 
            nann_metadata["metrics"]["latest"]["r2"]
        )
        self.__normalization_metric_direction_accuracy(
            stock_name, 
            gru_metadata["metrics"]["latest"]["direction_accuracy"], 
            lstm_metadata["metrics"]["latest"]["direction_accuracy"], 
            rfr_metadata["metrics"]["latest"]["direction_accuracy"], 
            nann_metadata["metrics"]["latest"]["direction_accuracy"]
        )
        self.__normalization_metric_mse(
            stock_name, 
            gru_metadata["metrics"]["latest"]["mse"], 
            lstm_metadata["metrics"]["latest"]["mse"], 
            rfr_metadata["metrics"]["latest"]["mse"], 
            nann_metadata["metrics"]["latest"]["mse"]
        )
        self.__normalization_metric_mae(
            stock_name, 
            gru_metadata["metrics"]["latest"]["mae"], 
            lstm_metadata["metrics"]["latest"]["mae"], 
            rfr_metadata["metrics"]["latest"]["mae"], 
            nann_metadata["metrics"]["latest"]["mae"]
        )

    def __count_weighted_r2(self, stock_name):
        self.gru_metric_score[stock_name]["r2"] *= self.r2_weight
        self.lstm_metric_score[stock_name]["r2"] *= self.r2_weight
        self.rfr_metric_score[stock_name]["r2"] *= self.r2_weight
        self.nann_metric_score[stock_name]["r2"] *= self.r2_weight

    def __count_weighted_accuracy(self, stock_name):
        self.gru_metric_score[stock_name]["accuracy"] *= self.accuracy_weight
        self.lstm_metric_score[stock_name]["accuracy"] *= self.accuracy_weight
        self.rfr_metric_score[stock_name]["accuracy"] *= self.accuracy_weight
        self.nann_metric_score[stock_name]["accuracy"] *= self.accuracy_weight

    def __count_weighted_mae(self, stock_name):
        self.gru_metric_score[stock_name]["mae"] *= self.mae_weight
        self.lstm_metric_score[stock_name]["mae"] *= self.mae_weight
        self.rfr_metric_score[stock_name]["mae"] *= self.mae_weight
        self.nann_metric_score[stock_name]["mae"] *= self.mae_weight
    
    def __count_weighted_mse(self, stock_name):
        self.gru_metric_score[stock_name]["mse"] *= self.mse_weight
        self.lstm_metric_score[stock_name]["mse"] *= self.mse_weight
        self.rfr_metric_score[stock_name]["mse"] *= self.mse_weight
        self.nann_metric_score[stock_name]["mse"] *= self.mse_weight

    def __count_weighted_score(self, stock_name):
        self.__count_weighted_r2(stock_name)
        self.__count_weighted_accuracy(stock_name)
        self.__count_weighted_mae(stock_name)
        self.__count_weighted_mse(stock_name)

    def __ranking_model_and_get_next_price(self, stock_name):
        ranking = {
            "gru" : self.gru_metric_score[stock_name]["r2"] + self.gru_metric_score[stock_name]["accuracy"] + self.gru_metric_score[stock_name]["mse"] + self.gru_metric_score[stock_name]["mae"],
            "lstm" : self.lstm_metric_score[stock_name]["r2"] + self.lstm_metric_score[stock_name]["accuracy"] + self.lstm_metric_score[stock_name]["mse"] + self.lstm_metric_score[stock_name]["mae"],
            "rfr" : self.rfr_metric_score[stock_name]["r2"] + self.rfr_metric_score[stock_name]["accuracy"] + self.rfr_metric_score[stock_name]["mse"] + self.rfr_metric_score[stock_name]["mae"],
            "nann" : self.nann_metric_score[stock_name]["r2"] + self.nann_metric_score[stock_name]["accuracy"] + self.nann_metric_score[stock_name]["mse"] + self.nann_metric_score[stock_name]["mae"]
        }

        sorted_ranking = sorted(
            ranking.items(),
            key=lambda x: x[1],
            reverse=True
        )

        self.best_model_predicted[stock_name] = sorted_ranking[0][0].upper()

        if sorted_ranking[0][0] == "gru":
            self.best_metadatas[stock_name] = self.gru_repository.metadata[stock_name]
            self.next_price[stock_name] = self.gru_repository.predict_next_day(stock_name)
        elif sorted_ranking[0][0] == "lstm":
            self.best_metadatas[stock_name] = self.lstm_repository.metadata[stock_name]
            self.next_price[stock_name] = self.lstm_repository.predict_next_day(stock_name)
        elif sorted_ranking[0][0] == "rfr":
            self.best_metadatas[stock_name] = self.rfr_repository.metadata[stock_name]
            self.next_price[stock_name] = self.rfr_repository.predict_next_day(stock_name)
        else:
            self.best_metadatas[stock_name] = self.nann_repository.metadata[stock_name]
            self.next_price[stock_name] = self.nann_repository.predict_next_day(stock_name)