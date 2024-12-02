import os
from flask import Flask, request, jsonify
from transformers import LlamaTokenizer, LlamaForCausalLM

app = Flask(__name__)

# Load Hugging Face Token from environment variables
hf_token = os.environ.get("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")

if not hf_token:
    raise ValueError("Hugging Face token (HF_TOKEN) not found in environment variables.")
    
# Load the LLaMA model and tokenizer with the token
print("Loading LLaMA model and tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=hf_token)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=hf_token)
print("Model loaded successfully!")

# Function to generate recommendations
def recommend_projects(input_tags, num_recommendations=3):
    inputs = tokenizer(input_tags, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=200, num_return_sequences=num_recommendations)
    recommendations = []
    for i, output in enumerate(outputs):
        decoded_text = tokenizer.decode(output, skip_special_tokens=True)
        recommendations.append({"title": f"Project {i+1}", "description": decoded_text.strip()})
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('tags', '')
    if not user_input:
        return jsonify({"error": "No input tags provided"}), 400
    
    recommendations = recommend_projects(user_input)
    return jsonify(recommendations)

@app.route('/')
def home():
    return "Welcome to the Project Recommender!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render uses the "PORT" environment variable
    app.run(host='0.0.0.0', port=port)
