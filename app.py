import time
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/heartbeat", methods = ["GET"])
def heartbeat():
    heartbeat = time.monotonic_ns()
    return {"heartbeat": heartbeat}, 200

if __name__ == "__main__":
    app.run(debug=True)
