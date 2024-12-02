from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Function to load projects from an Excel file
def load_projects(file_path):
    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return pd.DataFrame(columns=['id', 'title', 'description'])
    except Exception as e:
        print(f"Error loading file: {e}")
        return pd.DataFrame(columns=['id', 'title', 'description'])

# Load project ideas from an Excel file
projects = load_projects("project_ideas.xlsx")  # Replace with your file name

# Check if the file is loaded successfully
if projects.empty:
    print("No projects loaded. Please check the Excel file.")
else:
    # TF-IDF Vectorizer for content-based filtering
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(projects['description'])

# Recommendation function
def recommend_projects(input_tags, num_recommendations=3):
    try:
        if projects.empty:
            return [{"error": "No projects available for recommendations."}]
        
        input_vector = tfidf.transform([input_tags])
        cosine_similarities = linear_kernel(input_vector, tfidf_matrix).flatten()
        related_indices = cosine_similarities.argsort()[-num_recommendations:][::-1]
        recommendations = projects.iloc[related_indices]
        return recommendations[['id', 'title', 'description']].to_dict(orient='records')
    except Exception as e:
        print(f"Error during recommendation: {e}")
        return [{"error": "An error occurred during recommendation."}]

@app.route('/')
def home():
    return render_template('index.html')  # Ensure this file exists in the `templates` folder

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('tags', '').strip()
    if not user_input:
        return jsonify([{"error": "No tags provided. Please enter some tags."}])
    recommendations = recommend_projects(user_input)
    return jsonify(recommendations)

# Route to reload the Excel file dynamically (Optional)
@app.route('/reload', methods=['POST'])
def reload_projects():
    global projects, tfidf_matrix
    try:
        projects = load_projects("project_ideas.xlsx")  # Reload the Excel file
        if projects.empty:
            return "No projects found. Please check the Excel file."
        tfidf_matrix = tfidf.fit_transform(projects['description'])
        return "Projects reloaded successfully!"
    except Exception as e:
        print(f"Error reloading projects: {e}")
        return f"Error reloading projects: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
