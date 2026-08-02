import socketio
from models import PredictModel

class SocketClient:
    def __init__(self):
        self.is_news_updated = False
        self.is_orderbook_updated = False
        self.is_processed_news_updated = False
        self.is_stock_updated = False
        self.sio = socketio.Client(
            reconnection=False,
        )
        self.__register_events()

    def __register_events(self):
        @self.sio.event
        def connect():
            print("Connected")

        @self.sio.event
        def disconnect():
            print("Disconnected")

        @self.sio.on("news_updated")
        def news_updated(data):
            self.is_news_updated = True
            self.check_process_and_run()

        @self.sio.on("orderbook_updated")
        def orderbook_updated(data):
            self.is_orderbook_updated = True
            self.check_process_and_run()

        @self.sio.on("process_data_updated")
        def process_data_updated(data):
            self.is_processed_news_updated = True
            self.check_process_and_run()

        @self.sio.on("stock_updated")
        def stock_updated(data):
            self.is_stock_updated = True
            self.check_process_and_run()

    def check_process_and_run(self):
        if self.is_news_updated and self.is_orderbook_updated and self.is_processed_news_updated and self.is_stock_updated:
            model = PredictModel()
            model.analyze_all()
            next_price = model.next_price

            self.sio.emit(
                "prediction_updated",
                {
                    "data": next_price
                }
            )

            self.is_news_updated = False
            self.is_orderbook_updated = False
            self.is_processed_news_updated = False
            self.is_stock_updated = False

    def start(self):
        try:
            self.sio.connect(
                "http://127.0.0.1:5000",
                transports=["polling"]
            )
            self.sio.wait()
        except Exception as e:
            print("Error: ", e)