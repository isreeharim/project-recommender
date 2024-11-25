from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Sample dataset of project ideas
projects = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'title': [
        'AI Chatbot for Customer Service',
        'IoT-based Smart Home System',
        'Blockchain for Secure Voting',
        'E-commerce Recommendation System',
        'Healthcare Prediction with Machine Learning',
    ],
    'description': [
        'Build a chatbot using NLP techniques for automating customer interactions.',
        'Design a smart home system using IoT devices like sensors and actuators.',
        'Implement a secure voting system using blockchain technology.',
        'Create a recommendation system for e-commerce platforms using user data.',
        'Predict health conditions using machine learning and patient data.'
    ],
    'tags': ['AI', 'IoT', 'Blockchain', 'AI', 'Healthcare']
})

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

