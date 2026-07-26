import socketio

class SocketClient:
    def __init__(self):
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
            print(data)

        @self.sio.on("orderbook_updated")
        def orderbook_updated(data):
            print(data)

        @self.sio.on("process_data_updated")
        def process_data_updated(data):
            print(data)

        @self.sio.on("stock_updated")
        def stock_updated(data):
            print(data)

    def start(self):
        try:
            self.sio.connect(
                "http://127.0.0.1:5000",
                transports=["polling"]
            )
            self.sio.wait()
        except Exception as e:
            print("Error: ", e)