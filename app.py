from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

app = Flask(__name__)

# Function to load projects from the Excel file
def load_projects(file_path):
    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        raise Exception(f"Excel file '{file_path}' not found. Please ensure it exists.")

# Load project ideas from an Excel file
EXCEL_FILE = "project_ideas.xlsx"  # Replace with your actual file name
projects = load_projects(EXCEL_FILE)

# TF-IDF Vectorizer for content-based filtering
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(projects["description"])

# Recommendation function
def recommend_projects(input_tags, num_recommendations=3):
    input_vector = tfidf.transform([input_tags])
    cosine_similarities = linear_kernel(input_vector, tfidf_matrix).flatten()
    related_indices = cosine_similarities.argsort()[-num_recommendations:][::-1]
    recommendations = projects.iloc[related_indices]
    return recommendations[["id", "title", "description"]].to_dict(orient="records")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.form.get("tags")
    recommendations = recommend_projects(user_input)
    return jsonify(recommendations)

@app.route("/reload", methods=["POST"])
def reload_projects():
    global projects, tfidf_matrix
    try:
        projects = load_projects(EXCEL_FILE)
        tfidf_matrix = tfidf.fit_transform(projects["description"])
        return "Projects reloaded successfully!"
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    # Use port 5000 by default or set the PORT environment variable for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
