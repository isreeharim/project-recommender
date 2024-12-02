from flask import Flask, request, jsonify, render_template
import openai

app = Flask(__name__)

# Configure your OpenAI API key
openai.api_key = "sk-ijklmnopqrstuvwxijklmnopqrstuvwxijklmnop"  # Replace with your actual OpenAI API key

# Function to get recommendations from ChatGPT
def get_chatgpt_recommendations(tags, num_recommendations=3):
    try:
        # Construct a prompt for ChatGPT
        prompt = (
            f"Generate {num_recommendations} innovative project ideas based on the following tags: {tags}. "
            "Each idea should have a title and a brief description."
        )
        
        # Call the OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-003",  # Or use "gpt-4" if available
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )
        
        # Parse the response text
        output = response.choices[0].text.strip().split("\n\n")  # Split ideas
        recommendations = []
        for idea in output:
            if ": " in idea:
                title, description = idea.split(": ", 1)
                recommendations.append({"title": title.strip(), "description": description.strip()})
            else:
                recommendations.append({"title": idea.strip(), "description": ""})
        
        return recommendations
    except Exception as e:
        print(f"Error fetching ChatGPT recommendations: {e}")
        return [{"error": "Failed to generate recommendations. Please try again later."}]

@app.route('/')
def home():
    return render_template('index.html')  # Ensure this file exists in the `templates` folder

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('tags', '').strip()
    if not user_input:
        return jsonify([{"error": "No tags provided. Please enter some tags."}])
    
    recommendations = get_chatgpt_recommendations(user_input)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
