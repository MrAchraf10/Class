from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from CHATBOT import chat

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Welcome to the Flask app!"

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = chat(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug = True)
