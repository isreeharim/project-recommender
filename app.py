from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Function to load projects from an Excel file
def load_projects(file_path):
    return pd.read_excel(file_path)

# Load project ideas from an Excel file
projects = load_projects("project_ideas.xlsx")  # Replace with your file name

# TF-IDF Vectorizer for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(projects['description'])

# Recommendation function
def recommend_projects(input_tags, num_recommendations=3):
    input_vector = tfidf.transform([input_tags])
    cosine_similarities = linear_kernel(input_vector, tfidf_matrix).flatten()
    related_indices = cosine_similarities.argsort()[-num_recommendations:][::-1]
    recommendations = projects.iloc[related_indices]
    return recommendations[['id', 'title', 'description']].to_dict(orient='records')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('tags')
    recommendations = recommend_projects(user_input)
    return jsonify(recommendations)

# Route to reload the Excel file dynamically (Optional)
@app.route('/reload', methods=['POST'])
def reload_projects():
    global projects, tfidf_matrix
    projects = load_projects("project_ideas.xlsx")  # Reload the Excel file
    tfidf_matrix = tfidf.fit_transform(projects['description'])
    return "Projects reloaded successfully!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
