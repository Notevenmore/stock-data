from flask import Flask
from socket_client import SocketClient
from threading import Thread

app = Flask(__name__)

def start_socket_client():
    client = SocketClient()
    client.start()

@app.route("/")
def index():
    return "ML Server Running"

if __name__ == "__main__":
    socket_thread = Thread(
        target=start_socket_client,
        daemon=True
    )

    socket_thread.start()
    
    app.run(
        host="0.0.0.0",
        port=6000,
        debug=False,
        load_dotenv=True,
    )