from repositories import PreprocessingRepository

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import StandardScaler

import os
import joblib
import json
from datetime import datetime

import pandas as pd
import numpy as np

class LSTMRepository(PreprocessingRepository):
    def __init__(self):
        super().__init__()
        self.standard_scaler = {}
        self.y_standard_scaler = {}
        self.model = {}
        self.result = {}
        self.features = {}
        self.metadata = {}
        self.is_scaler_already_built = {}
        self.train_size = {}
        self.test_size = {}

        self.init_standard_scaler_stocks()

    def init_standard_scaler_stocks(self):
        for _, stock in self.stock_list.iterrows():
            self.__init_standard_scaler_stock(stock['code'])

    def __init_standard_scaler_stock(self, stock_name):
        path = f"data/scaled/lstm/{stock_name}"
        x_scaler_path = f"{path}/x.joblib"
        y_scaler_path = f"{path}/y.joblib"

        if os.path.exists(x_scaler_path) and os.path.exists(y_scaler_path):
            self.standard_scaler[stock_name] = joblib.load(x_scaler_path)
            self.y_standard_scaler[stock_name] = joblib.load(y_scaler_path)
            self.is_scaler_already_built[stock_name] = True
        else:
            self.standard_scaler[stock_name] = StandardScaler()
            self.y_standard_scaler[stock_name] = StandardScaler()

            self.is_scaler_already_built[stock_name] = False

    def workflow_analyze_stocks(self):
        for _, stock in self.stock_list.iterrows():
            self.__workflow_analyze_stock(stock['code'])

    def __workflow_analyze_stock(self, stock_name):
        self.features[stock_name] = list(self.x_train[stock_name].columns)
        self.__scaled_data(stock_name)

        self.x_train[stock_name], self.y_train[stock_name] = self.create_sequences(self.x_train[stock_name], self.y_train[stock_name], window_size=10)
        self.x_test[stock_name], self.y_test[stock_name] = self.create_sequences(self.x_test[stock_name], self.y_test[stock_name], window_size=10)

        self.train_size[stock_name] = len(self.x_train[stock_name])
        self.test_size[stock_name] = len(self.x_test[stock_name])

        self.__run_model(stock_name)

    def __scaled_data(self, stock_name):
        if self.is_scaler_already_built[stock_name]:
            self.x_train[stock_name][self.scale_columns] = self.standard_scaler[stock_name].transform(self.x_train[stock_name][self.scale_columns])
            self.x_test[stock_name][self.scale_columns] = self.standard_scaler[stock_name].transform(self.x_test[stock_name][self.scale_columns])

            self.y_train[stock_name] = pd.Series(self.y_standard_scaler[stock_name].transform(self.y_train[stock_name].values.reshape(-1, 1)).flatten())
            self.y_test[stock_name] = pd.Series(self.y_standard_scaler[stock_name].transform(self.y_test[stock_name].values.reshape(-1, 1)).flatten())
        else:
            self.x_train[stock_name][self.scale_columns] = self.standard_scaler[stock_name].fit_transform(self.x_train[stock_name][self.scale_columns])
            self.x_test[stock_name][self.scale_columns] = self.standard_scaler[stock_name].transform(self.x_test[stock_name][self.scale_columns])

            self.y_train[stock_name] = pd.Series(self.y_standard_scaler[stock_name].fit_transform(self.y_train[stock_name].values.reshape(-1, 1)).flatten())
            self.y_test[stock_name] = pd.Series(self.y_standard_scaler[stock_name].transform(self.y_test[stock_name].values.reshape(-1, 1)).flatten())

            path = f"data/scaled/lstm/{stock_name}"
            os.makedirs(path, exist_ok=True)
            joblib.dump(self.standard_scaler[stock_name], f"{path}/x.joblib")
            joblib.dump(self.y_standard_scaler[stock_name], f"{path}/y.joblib")
            self.is_scaler_already_built[stock_name] = True

    def create_sequences(self, X, y, window_size=30):
        xs = []
        ys = []

        for i in range(len(X) - window_size):
            xs.append(
                X.iloc[i:i+window_size].values
            )

            ys.append(
                y.iloc[i+window_size]
            )

        return np.array(xs), np.array(ys)

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
        metadata_path = f"{path}/lstm-metadata.json"
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
                "model": "LSTM",
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
        metadata_path = f"{path}/lstm-metadata.json"
        
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
        model_path = f"{path}/lstm.keras"

        if os.path.exists(model_path):
            self.model[stock_name] = load_model(model_path)
        else: 
            self.model[stock_name] = Sequential([
                LSTM(
                    32,
                    return_sequences=False,
                    input_shape=(
                        self.x_train[stock_name].shape[1],
                        self.x_train[stock_name].shape[2]
                    )
                ),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1)
            ])

            self.model[stock_name].compile(
                optimizer="adam",
                loss=Huber(),
                metrics=[
                    "mae"
                ]
            )

            self.model[stock_name].fit(
                self.x_train[stock_name], 
                self.y_train[stock_name],
                epochs=100,
                batch_size=32,
                validation_data=(
                    self.x_test[stock_name],
                    self.y_test[stock_name]
                )
            )

            os.makedirs(path, exist_ok=True)
            self.model[stock_name].save(f"{path}/lstm.keras")

    def __predict_data(self, stock_name):
        pred = self.model[stock_name].predict(self.x_test[stock_name])
        pred = self.y_standard_scaler[stock_name].inverse_transform(pred)
        actual = self.y_standard_scaler[stock_name].inverse_transform(self.y_test[stock_name].reshape(-1, 1))

        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        accuracy = accuracy_score(
            (actual.flatten() > 0).astype(int),
            (pred.flatten() > 0).astype(int)
        )

        result = pd.DataFrame({
            "actual": actual.flatten(),
            "prediction": pred.flatten(),
            "actual_direction": (actual.flatten() > 0).astype(int),
            "prediction_direction": (pred.flatten() > 0).astype(int)
        })

        return mse, mae, r2, accuracy, result

    def predict_next_day(self, stock_name):
        latest_data = self.x_test[stock_name][-1]
        latest_data = np.expand_dims(latest_data, axis=0)

        pred = self.model[stock_name].predict(latest_data)
        pred = self.y_standard_scaler[stock_name].inverse_transform(pred)

        return float(pred[0][0])