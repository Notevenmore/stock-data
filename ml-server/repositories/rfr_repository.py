from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

import os
import joblib
import json
from datetime import datetime

import pandas as pd
import numpy as np

class RFRRepository:
    def __init__(self, pre_processing_repository):
        self.x_train = pre_processing_repository.x_train.copy()
        self.x_test= pre_processing_repository.x_test.copy()
        self.y_train = pre_processing_repository.y_train.copy()
        self.y_test = pre_processing_repository.y_test.copy()
        self.stock_list = pre_processing_repository.stock_list
        self.model = {}
        self.result = {}
        self.features = {}
        self.metadata = {}
        self.train_size = {}
        self.test_size = {}

    def workflow_analyze_stocks(self):
        for _, stock in self.stock_list.iterrows():
            self.__workflow_analyze_stock(stock['code'])

    def __workflow_analyze_stock(self, stock_name):
        self.features[stock_name] = list(self.x_train[stock_name].columns)

        self.train_size[stock_name] = len(self.x_train[stock_name])
        self.test_size[stock_name] = len(self.x_test[stock_name])
        
        self.__run_model(stock_name)

    def __run_model(self, stock_name):
        self.__training_model(stock_name)
        mse, mae, r2, accuracy, result = self.__predict_data(stock_name)
        self.result[stock_name] = {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "result": result,
            "accuracy": accuracy
        }

        self.__save_metadata(stock_name, mse, mae, r2, accuracy, result)

    def __save_metadata(self, stock_name, mse, mae, r2, accuracy, result):
        path = f"data/models/{stock_name}"
        metadata_path = f"{path}/rfr-metadata.json"
        os.makedirs(path, exist_ok=True)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as file:
                metadata = json.load(file)

            metadata["result"] = result.to_dict(orient="records")
            metadata["metrics"]["latest"] = {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "direction_accuracy": float(accuracy)
            }
            count = metadata["evaluation_count"]

            metadata["metrics"]["average"]["mse"] = float(
                ((metadata["metrics"]["average"]["mse"] * count) + mse)
                / (count + 1)
            )

            metadata["metrics"]["average"]["mae"] = float(
                ((metadata["metrics"]["average"]["mae"] * count) + mae)
                / (count + 1)
            )

            metadata["metrics"]["average"]["r2"] = float(
                ((metadata["metrics"]["average"]["r2"] * count) + r2)
                / (count + 1)
            )

            metadata["metrics"]["average"]["direction_accuracy"] = float(
                ((metadata["metrics"]["average"]["direction_accuracy"] * count) + accuracy)
                / (count + 1)
            )

            metadata["evaluation_count"] += 1
            metadata["latest_evaluated_at"] = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        else:
            metadata = {
                "stock": stock_name,
                "model": "RFR",
                "window_size": 10,
                "dataset": {
                    "train_size": self.train_size[stock_name],
                    "test_size": self.test_size[stock_name],
                    "feature_count": len(self.features[stock_name]), 
                    "features": list(self.features[stock_name]),
                },
                "trained_at": datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
                "latest_evaluated_at": datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
                "evaluation_count": 1,
                "metrics": {
                    "latest": {
                        "mse": float(mse),
                        "mae": float(mae),
                        "r2": float(r2),
                        "direction_accuracy": float(accuracy)
                    },
                    "average": {
                        "mse": float(mse),
                        "mae": float(mae),
                        "r2": float(r2),
                        "direction_accuracy": float(accuracy)
                    },
                },
                "result": result.to_dict(orient="records")
            }

        with open(metadata_path, "w") as file:
            json.dump(
                metadata,
                file,
                indent=4
            )

    def load_metadatas(self):
        for _, stock in self.stock_list.iterrows():
            self.__load_metadata(stock['code'])

    def __load_metadata(self, stock_name):
        path = f"data/models/{stock_name}"
        metadata_path = f"{path}/rfr-metadata.json"

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as file:
                metadata = json.load(file)
                self.metadata[stock_name] = metadata
                self.result[stock_name] = {
                    "mse": metadata["metrics"]["latest"]["mse"],
                    "mae": metadata["metrics"]["latest"]["mae"],
                    "r2": metadata["metrics"]["latest"]["r2"],
                    "result": pd.DataFrame(metadata["result"]),
                    "accuracy": metadata["metrics"]["latest"]["direction_accuracy"]
                }
        else:
            self.metadata[stock_name] = None

    def __training_model(self, stock_name):
        path = f"data/models/{stock_name}"
        model_path = f"{path}/rfr.joblib"

        if os.path.exists(model_path):
            self.model[stock_name] = joblib.load(model_path)
        else: 
            self.model[stock_name] = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=3,
                random_state=0,
                n_jobs=1,
                max_features="log2",
                criterion="friedman_mse",
                min_samples_split=5,
                min_weight_fraction_leaf=0,
                min_impurity_decrease=0.9,
                oob_score=True,
                warm_start=True,
            )

            self.model[stock_name].fit(
                self.x_train[stock_name]   ,
                self.y_train[stock_name]
            )

            os.makedirs(path, exist_ok=True)
            joblib.dump(self.model[stock_name], model_path)

    def __predict_data(self, stock_name):
        pred = self.model[stock_name].predict(self.x_test[stock_name])

        mse = mean_squared_error(self.y_test[stock_name], pred)
        mae = mean_absolute_error(self.y_test[stock_name], pred)
        r2 = r2_score(self.y_test[stock_name], pred)
        
        accuracy = accuracy_score(
            (self.y_test[stock_name] > 0).astype(int),
            (pred > 0).astype(int)
        )

        result = pd.DataFrame({
            "actual": self.y_test[stock_name],
            "prediction": pred,
            "actual_direction": (self.y_test[stock_name] > 0).astype(int),
            "prediction_direction": (pred > 0).astype(int)
        })

        return mse, mae, r2, accuracy, result

    def predict_next_day(self, stock_name):
        latest_data = self.x_test[stock_name][-1:]

        pred = self.model[stock_name].predict(latest_data)

        return float(pred[0])