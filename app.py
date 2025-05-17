import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os.path
import functools

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key_change_in_production")

# Admin credentials
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Initialize TF-IDF vectorizer for text embeddings
vectorizer = TfidfVectorizer()

# Vector database setup
DB_PATH = "vector_db"
VECTORIZER_FILE = os.path.join(DB_PATH, "vectorizer.pkl")
TEXT_DATA_FILE = os.path.join(DB_PATH, "text_data.pkl")
VECTORS_FILE = os.path.join(DB_PATH, "vectors.pkl")

# Create DB directory if it doesn't exist
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# Initialize or load vectorizer and data
if os.path.exists(VECTORIZER_FILE) and os.path.exists(TEXT_DATA_FILE) and os.path.exists(VECTORS_FILE):
    # Load existing vectorizer and data
    with open(VECTORIZER_FILE, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(TEXT_DATA_FILE, 'rb') as f:
        text_data = pickle.load(f)
    with open(VECTORS_FILE, 'rb') as f:
        vectors = pickle.load(f)
else:
    # Create new empty data list and initialize vectors as None
    text_data = []
    vectors = None

def add_to_vector_db(text):
    """Add text to the vector database"""
    global vectorizer, text_data, vectors
    
    # Add text to the data list
    text_data.append(text)
    
    # Fit vectorizer on all texts and transform to get vectors
    vectors = vectorizer.fit_transform(text_data)
    
    # Save updated vectorizer and data
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(TEXT_DATA_FILE, 'wb') as f:
        pickle.dump(text_data, f)
    with open(VECTORS_FILE, 'wb') as f:
        pickle.dump(vectors, f)
    
    return True

def search_vector_db(query, k=5):
    """Search for similar texts in the vector database"""
    global vectorizer, text_data, vectors
    
    if not text_data or vectors is None:
        return []
    
    # Transform query to vector using the same vectorizer
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity between query and all stored vectors
    similarities = cosine_similarity(query_vector, vectors).flatten()
    
    # Get indices of top k similar texts
    top_indices = similarities.argsort()[-k:][::-1]
    
    # Get corresponding texts
    results = [text_data[idx] for idx in top_indices]
    
    return results

def get_gemini_response(prompt, context=""):
    """Get response from Gemini Flash 2.0 model"""
    try:
        # Configure the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        # Use Gemini 1.5 Pro model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )
        
        # Prepare the prompt with context if available
        if context:
            full_prompt = f"Context information: {context}\n\nUser question: {prompt}\n\nPlease answer based on the context provided."
        else:
            full_prompt = prompt
        
        # Generate response
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/')
def index():
    """Render the chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"response": "Please enter a message."})
    
    # Search for relevant context in vector DB
    context_texts = search_vector_db(user_message)
    context = "\n".join(context_texts) if context_texts else ""
    
    # Get response from Gemini
    response = get_gemini_response(user_message, context)
    
    # Process response to make it more suitable for typewriter effect
    # Remove excessive newlines and spaces to improve the typing animation
    processed_response = response.strip()
    
    return jsonify({"response": processed_response})

# Admin login required decorator
def admin_login_required(func):
    @functools.wraps(func)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('You need to be logged in as an admin to access this page.', 'danger')
            return redirect(url_for('admin_login'))
        return func(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.pop('admin_logged_in', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
@admin_login_required
def admin_dashboard():
    """Admin dashboard"""
    return render_template('admin_dashboard.html', text_data=text_data)

@app.route('/admin/add_data', methods=['GET', 'POST'])
@admin_login_required
def add_data():
    """Handle data addition requests (admin only)"""
    if request.method == 'POST':
        data = request.form.get('data', '')
        
        if not data:
            flash('Please enter some data.', 'warning')
            return render_template('add_data.html')
        
        # Add data to vector DB
        success = add_to_vector_db(data)
        
        if success:
            flash('Data added successfully!', 'success')
        else:
            flash('Failed to add data.', 'danger')
    
    return render_template('add_data.html')

if __name__ == '__main__':
    app.run(debug=True)
