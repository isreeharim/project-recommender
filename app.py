from flask import Flask, request, render_template
import requests
import json

app = Flask(__name__)

# Google Gemini API Endpoint and Key
API_KEY = "AIzaSyCQSNlgod8yCyBfFMNf40WovBNMaBBmfpY"
API_URL = "https://gemini.googleapis.com/v1/models/text:generate"  # Replace with the actual API endpoint if different

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.form.get("query")

    # Google Gemini API call setup
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Define the payload for the API call to Google Gemini (or PaLM)
    payload = {
        "prompt": f"Suggest project ideas based on the following interests: {user_input}",
        "max_tokens": 200,
        "temperature": 0.7
    }

    # Send POST request to Google Gemini API
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        recommendations = response.json().get("choices", [{}])[0].get("text", "").split("\n")
    else:
        recommendations = ["Error: Could not fetch recommendations from Google Gemini."]
    
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
