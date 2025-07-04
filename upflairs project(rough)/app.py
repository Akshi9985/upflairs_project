from flask import Flask, render_template, request, jsonify
import random
import json
import joblib

model = joblib.load('chatbot_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

with open('intent.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

app = Flask(__name__)

def predict_intent(text, model, vectorizer, label_encoder):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    return label_encoder.inverse_transform(prediction)[0]

def get_response(intent_tag, intents_data):
    for intent in intents_data['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form["msg"]
    print("User input:", user_input)

    intent = predict_intent(user_input, model, vectorizer, label_encoder)
    print("Predicted intent:", intent)

    response = get_response(intent, data)
    print("Selected response:", response)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
