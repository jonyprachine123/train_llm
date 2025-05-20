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
import json
import uuid

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
PRODUCT_DESC_FILE = os.path.join(DB_PATH, "product_descriptions.json")
CHAT_MEMORY_FILE = os.path.join(DB_PATH, "chat_memory.json")

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

# Initialize or load product descriptions
if os.path.exists(PRODUCT_DESC_FILE):
    with open(PRODUCT_DESC_FILE, 'r', encoding='utf-8') as f:
        product_descriptions = json.load(f)
        
        # Create simplified product name variants for easier matching
        for product in product_descriptions:
            # Create a simplified version of the product name (without special characters)
            simple_name = ''.join(c for c in product['name'] if c.isalnum() or c.isspace())
            product['simple_name'] = simple_name
            
            # Create a list of key words from the product name
            words = [w for w in product['name'].split() if len(w) > 3]
            product['keywords'] = words
else:
    product_descriptions = []
    
# Initialize or load chat memory
if os.path.exists(CHAT_MEMORY_FILE):
    with open(CHAT_MEMORY_FILE, 'r', encoding='utf-8') as f:
        chat_memory = json.load(f)
else:
    chat_memory = {}
    
# Function to extract product names from text
def extract_product_names(text, product_descriptions):
    """Extract product names mentioned in text by comparing with known products"""
    mentioned_products = []
    
    # Check for exact product name mentions
    for product in product_descriptions:
        product_name = product['name'].lower()
        # Check for full product name or partial matches for multi-word product names
        if product_name in text.lower() or any(word in text.lower() for word in product_name.split() if len(word) > 3):
            mentioned_products.append({
                'name': product['name'],
                'last_mentioned': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    return mentioned_products

# Common Bangla/Bengali names to avoid confusing with products
common_bangla_names = [
    'emon', 'ismail', 'mohammad', 'mohammed', 'muhammad', 'ahmed', 'rahman', 'khan', 'hasan', 'hassan',
    'karim', 'rahim', 'islam', 'uddin', 'ali', 'hossain', 'hussain', 'ahmed', 'begum', 'khatun',
    'akter', 'akther', 'jahan', 'sultana', 'miah', 'mia', 'chowdhury', 'roy', 'das', 'dey',
    'sarkar', 'saha', 'sen', 'poddar', 'mondal', 'sheikh', 'shaikh', 'talukder', 'tarafdar', 'biswas'
]

# Function to detect if a new product is being mentioned (context switch)
def detect_product_switch(message, product_descriptions, current_product=None):
    """Detect if user is asking about a new product (switching context)"""
    # Extract potential product names from the message
    new_products = []
    
    # Check if the message is likely about a person rather than a product
    message_words = message.lower().split()
    person_name_indicators = ['name', 'person', 'boy', 'girl', 'man', 'woman', 'age', 'old', 'young', 
                             'boyos', 'bhai', 'apu', 'chele', 'meye', 'lok', 'manush', 'betar', 'meyer']
    
    # Check for common Bangla names in the message
    has_person_name = any(name in message.lower() for name in common_bangla_names)
    
    # Check for person-related question indicators
    is_person_question = any(indicator in message.lower() for indicator in person_name_indicators)
    
    # If this appears to be a question about a person rather than a product, don't switch context
    if has_person_name and is_person_question:
        return False, None
    
    # Otherwise, check for product names
    for product in product_descriptions:
        product_name = product['name'].lower()
        product_words = product_name.split()
        
        # Check for exact matches or significant partial matches
        if product_name in message.lower():
            new_products.append(product['name'])
        elif len(product_words) > 1:
            # For multi-word product names, check if any significant word is mentioned
            for word in product_words:
                if len(word) > 3 and word in message.lower() and word not in common_bangla_names:
                    new_products.append(product['name'])
                    break
    
    # If we found products and they're different from the current product, it's a switch
    if new_products and (current_product is None or not any(p.lower() == current_product.lower() for p in new_products)):
        return True, new_products[0]  # Return the first detected new product
    
    return False, None

# Function to check if a message is asking about a product without naming it
def is_product_followup_question(message):
    """Check if a message is likely a follow-up question about a product"""
    # English follow-up indicators
    english_indicators = ['how', 'what', 'when', 'where', 'why', 'who', 'which', 'can', 'does', 'is', 'are', 'was', 'were']
    
    # Bangla/Benglish follow-up indicators
    bangla_indicators = [
        'ata', 'eta', 'oita', 'ei', 'oi', 'sheta', 'kemon', 'kivabe', 'koto', 'kobe', 'kothay', 'ki', 'ke',
        'korte', 'hoi', 'hoy', 'lage', 'lagbe', 'dorkar', 'use', 'price', 'dam', 'daam', 'kharap', 'bhalo',
        'valo', 'khaite', 'khete', 'dekhte', 'shundar', 'sundor', 'shundor', 'niye', 'theke', 'banan', 'banano',
        'banate', 'baniye', 'kora', 'korar', 'kori', 'korchi', 'korbo', 'hobe', 'hobe na', 'pari', 'parbo',
        'parbo na', 'jabe', 'jabe na', 'jabo', 'jabo na', 'jete', 'jete pari', 'jete parbo', 'jete parbo na',
        'kete', 'kabo'
    ]
    
    # Common Bangla pronouns and demonstratives that might refer to products
    bangla_pronouns = ['ata', 'eta', 'oita', 'eita', 'sheta', 'ei', 'oi', 'ei jinish', 'oi jinish']
    
    # Check if message contains any follow-up indicators
    message_lower = message.lower()
    has_english_indicator = any(indicator in message_lower for indicator in english_indicators)
    has_bangla_indicator = any(indicator in message_lower for indicator in bangla_indicators)
    has_bangla_pronoun = any(pronoun in message_lower for pronoun in bangla_pronouns)
    
    # Check for common Bangla question patterns
    bangla_question_patterns = [
        'kivabe', 'kemon', 'koto', 'ki', 'ke', 'kobe', 'kothay',
        'korte hoi', 'korte hoy', 'use korte', 'lagbe', 'lage',
        'kivabe use', 'kivabe korte', 'kemon kore',
        'kivabe kete', 'kivabe kabo', 'kete hoi', 'kete hoy',
        'kabo kivabe', 'korbo kivabe', 'kete kabo', 'korte kabo'
    ]
    
    has_bangla_question = any(pattern in message_lower for pattern in bangla_question_patterns)
    
    # Check for short questions (likely follow-ups)
    is_short_question = len(message.split()) < 6
    
    # Return true if any of the indicators are present
    return has_english_indicator or has_bangla_indicator or has_bangla_pronoun or has_bangla_question or (is_short_question and ('?' in message))

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

def get_gemini_response(prompt, context="", language_instruction="", recent_product=None, product_descriptions=[]):
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
        
        # Base instructions for all queries
        context_instruction = """CRITICAL INSTRUCTIONS:
1. Never mention previous conversations or use phrases like 'আমাদের আগের আলোচনায়' (in our previous conversation) or 'আপনি আগে জিজ্ঞাসা করেছেন' (you asked earlier).
2. Answer the current question directly without referencing chat history.
3. NEVER mention products that aren't directly related to the current question.
4. If the question is about a company, organization, person, or general topic, DO NOT mention any product names in your response."""
        
        # Check for company/organization names in the prompt
        company_indicators = ['limited', 'ltd', 'company', 'organization', 'corporation', 'inc', 'kompani', 
                             'business', 'firm', 'enterprise', 'office', 'workplace', 'job', 'choti', 'leave', 
                             'holiday', 'chutti', 'beton', 'salary', 'employee', 'staff', 'kormi', 'kormocharee']
        
        has_company_topic = any(indicator in prompt.lower() for indicator in company_indicators)
        
        # Check if the prompt is asking about a person rather than a product
        person_indicators = ['name', 'person', 'boy', 'girl', 'man', 'woman', 'age', 'old', 'young',
                           'boyos', 'bhai', 'apu', 'chele', 'meye', 'lok', 'manush', 'betar', 'meyer']
        common_bangla_names = ['emon', 'ismail', 'mohammad', 'ahmed', 'rahman', 'khan', 'hasan']
        
        is_person_question = any(indicator in prompt.lower() for indicator in person_indicators)
        has_person_name = any(name in prompt.lower() for name in common_bangla_names)
        
        # If this is likely a question about a company or organization
        if has_company_topic:
            context_instruction += "\n\nCRITICAL: This question is about a COMPANY or ORGANIZATION. Do NOT mention ANY products in your response. Focus ONLY on answering the specific question about the company/organization."
            recent_product = None  # Clear product context for company questions
        # If this is likely a question about a person, override product context
        elif is_person_question and has_person_name:
            context_instruction += "\n\nCRITICAL: The user is asking about a PERSON, not a product. Do NOT mention ANY products in your response. This is a question about a person's details (like age, education, etc.)."
            recent_product = None  # Clear product context for person questions
        # Check if this is a direct product question (product name in prompt)
        elif product_descriptions:
            # Find the product being asked about and its description
            product_info = None
            
            # First check for exact product name matches
            for product in product_descriptions:
                if product['name'].lower() in prompt.lower():
                    recent_product = product['name']
                    product_info = product
                    break
            
            # If no exact match, check for key words (like "Zinova")
            if not product_info:
                for product in product_descriptions:
                    # Check for product keywords in the prompt
                    if any(word.lower() in prompt.lower() for word in product.get('keywords', [])):
                        recent_product = product['name']
                        product_info = product
                        break
                    # Special check for "Zinova" which might be in a product name
                    elif 'zinova' in product['name'].lower() and 'zinova' in prompt.lower():
                        recent_product = product['name']
                        product_info = product
                        break
                    
            # Add context for the direct product question with product details
            if product_info:
                product_description = product_info.get('description', '')
                context_instruction += f"\n\nIMPORTANT: The user is directly asking about '{recent_product}'. Here is the product information:\n\nProduct Name: {product_info['name']}\nDescription: {product_description}"
            else:
                context_instruction += f"\n\nIMPORTANT: The user is directly asking about '{recent_product}'. Provide information about this product."
            
        # Only if we're confident this is a follow-up product question, add product context
        elif recent_product and (recent_product.lower() in prompt.lower() or any(word in prompt.lower() for word in ['ata', 'eta', 'oita', 'eita', 'sheta', 'ei', 'oi', 'kivabe', 'korte', 'hoi', 'hoy', 'korbo', 'use', 'kabo', 'kete'])):
            # Check if the prompt contains Bangla indicators
            bangla_indicators = ['ata', 'eta', 'oita', 'eita', 'sheta', 'ei', 'oi', 'kivabe', 'korte', 'hoi', 'hoy', 'korbo', 'use', 'kabo', 'kete']
            has_bangla_indicator = any(word in prompt.lower() for word in bangla_indicators)
            
            # Find the product information for the recent product
            product_info = None
            for product in product_descriptions:
                if product['name'].lower() == recent_product.lower() or recent_product.lower() in product['name'].lower():
                    product_info = product
                    break
            
            # If the prompt has Bangla indicators, it's likely a follow-up
            if has_bangla_indicator and product_info:
                product_description = product_info.get('description', '')
                context_instruction += f"\n\nCRITICAL: The user is asking a follow-up question about '{recent_product}'. Here is the product information:\n\nProduct Name: {product_info['name']}\nDescription: {product_description}\n\nAnswer the user's specific question about how to use or what to do with this product based on this information. DO NOT apologize for not having information - use the product details provided."
            else:
                context_instruction += f"\n\nIMPORTANT: The user is asking about the product '{recent_product}'. You MUST mention '{recent_product}' explicitly in your response and focus ONLY on this product."
        
        # Prepare the prompt with context if available
        if context:
            if language_instruction:
                full_prompt = f"Information to help answer the question:\n{context}\n\n{context_instruction}\n\nCurrent user question: {prompt}\n\n{language_instruction}\n\nAnswer directly without mentioning previous conversations. Be specific and detailed in your response. ONLY mention products if they are directly relevant to the current question."
            else:
                full_prompt = f"Information to help answer the question:\n{context}\n\n{context_instruction}\n\nCurrent user question: {prompt}\n\nAnswer directly without mentioning previous conversations. Be specific and detailed in your response. ONLY mention products if they are directly relevant to the current question."
        else:
            if language_instruction:
                full_prompt = f"{context_instruction}\n\n{prompt}\n\n{language_instruction}\n\nAnswer directly without mentioning previous conversations. Be specific and detailed in your response. ONLY mention products if they are directly relevant to the current question."
            else:
                full_prompt = f"{context_instruction}\n\n{prompt}\n\nAnswer directly without mentioning previous conversations. Be specific and detailed in your response. ONLY mention products if they are directly relevant to the current question."
        
        # Generate response
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route('/')
def index():
    """Render the chat interface"""
    # Generate a session ID if not already present
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    global chat_memory
    
    data = request.json
    user_message = data.get('message', '')
    language_preference = data.get('language', 'english')
    
    if not user_message:
        return jsonify({"response": "Please enter a message."})
    
    # Get session ID from session or create a new one
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    # Initialize or get chat memory for this session
    if session_id not in chat_memory:
        chat_memory[session_id] = {
            'messages': [],
            'recent_products': []
        }
    elif not isinstance(chat_memory[session_id], dict):
        # Convert old format to new format if needed
        chat_memory[session_id] = {
            'messages': chat_memory[session_id] if isinstance(chat_memory[session_id], list) else [],
            'recent_products': []
        }
    
    # Get previous conversation context (last 5 exchanges)
    conversation_history = chat_memory[session_id]['messages'][-5:] if chat_memory[session_id]['messages'] else []
    
    # Format conversation history for context
    conversation_parts = []
    for msg in conversation_history:
        if isinstance(msg, dict) and 'user' in msg and 'ai' in msg:
            conversation_parts.append(f"User: {msg['user']}")
            conversation_parts.append(f"AI: {msg['ai']}")
    
    conversation_context = "\n".join(conversation_parts)
    
    # Get recently mentioned products
    recent_products = []
    if session_id in chat_memory and isinstance(chat_memory[session_id], dict) and 'recent_products' in chat_memory[session_id]:
        recent_products = chat_memory[session_id]['recent_products']
    
    # Get product details for recently mentioned products
    product_context = ""
    most_recent_product = None
    
    if recent_products:
        # Sort products by last_mentioned time if available
        if isinstance(recent_products[0], dict) and 'last_mentioned' in recent_products[0]:
            sorted_products = sorted(recent_products, key=lambda x: x.get('last_mentioned', ''), reverse=True)
            product_names = [p['name'] for p in sorted_products]
            if sorted_products:
                most_recent_product = sorted_products[0]['name']
        else:
            # Legacy format - just use the list as is
            product_names = recent_products
            if product_names:
                most_recent_product = product_names[0]
        
        product_details = []
        for product_name in product_names:
            matching_products = [p for p in product_descriptions if p['name'] == product_name]
            if matching_products:
                product = matching_products[0]
                product_details.append(f"Product: {product['name']}\nDescription: {product['description']}")
        
        if product_details:
            product_context = "Recently discussed products:\n" + "\n\n".join(product_details)
    
    # Check for different types of questions that should clear product context
    
    # Company/organization indicators
    company_indicators = ['limited', 'ltd', 'company', 'organization', 'corporation', 'inc', 'kompani', 
                         'business', 'firm', 'enterprise', 'office', 'workplace', 'job', 'choti', 'leave', 
                         'holiday', 'chutti', 'beton', 'salary', 'employee', 'staff', 'kormi', 'kormocharee']
    
    # Person indicators
    person_indicators = ['name', 'person', 'boy', 'girl', 'man', 'woman', 'age', 'old', 'young',
                       'boyos', 'bhai', 'apu', 'chele', 'meye', 'lok', 'manush', 'betar', 'meyer']
    common_bangla_names = ['emon', 'ismail', 'mohammad', 'ahmed', 'rahman', 'khan', 'hasan']
    
    # Check for different types of questions
    has_company_topic = any(indicator in user_message.lower() for indicator in company_indicators)
    is_person_question = any(indicator in user_message.lower() for indicator in person_indicators)
    has_person_name = any(name in user_message.lower() for name in common_bangla_names)
    
    # Clear product context for non-product questions
    if has_company_topic or (is_person_question and has_person_name):
        is_followup = False
        most_recent_product = None  # Clear product context for non-product questions
    else:
        # First check if this is a direct product question
        direct_product_question = False
        
        # Check for exact product name matches
        for product in product_descriptions:
            if product['name'].lower() in user_message.lower():
                most_recent_product = product['name']
                direct_product_question = True
                is_followup = False
                break
        
        # If no exact match, check for key words (like "Zinova")
        if not direct_product_question:
            for product in product_descriptions:
                # Check for product keywords in the message
                if any(word.lower() in user_message.lower() for word in product.get('keywords', [])):
                    most_recent_product = product['name']
                    direct_product_question = True
                    is_followup = False
                    break
                # Special check for "Zinova" which might be in a product name
                elif 'zinova' in product['name'].lower() and 'zinova' in user_message.lower():
                    most_recent_product = product['name']
                    direct_product_question = True
                    is_followup = False
                    break
        
        # If not a direct product question, check if switching to a new product
        if not direct_product_question:
            is_product_switch, new_product = detect_product_switch(user_message, product_descriptions, most_recent_product)
            
            # If user is asking about a new product, update most_recent_product
            if is_product_switch and new_product:
                most_recent_product = new_product
                is_followup = False  # Not a follow-up if switching to a new product
            else:
                # Check if this is a follow-up question about the previous product
                is_followup = False
                if most_recent_product and most_recent_product.lower() not in user_message.lower():
                    # Short messages are likely follow-ups
                    if len(user_message.split()) < 6:
                        is_followup = True
                    # Messages with question indicators are likely follow-ups
                    elif is_product_followup_question(user_message):
                        is_followup = True
                    # Bangla messages with 'ata', 'eta', etc. are almost certainly follow-ups
                    elif any(word in user_message.lower() for word in ['ata', 'eta', 'oita', 'eita', 'sheta', 'ei', 'oi', 'kivabe', 'korte', 'hoi', 'hoy']):
                        is_followup = True
    
    # Search for relevant context in vector DB
    context_texts = search_vector_db(user_message)
    
    # Detect if message is in Benglish (Bengali written with English characters)
    benglish_indicators = [
        'ami', 'tumi', 'apni', 'ke', 'ki', 'kemon', 'ache', 'acho', 'kore', 'korchi', 'korbo',
        'bhalo', 'bhalobasa', 'bhai', 'bon', 'baba', 'ma', 'khabar', 'khabo', 'jabo', 'asbo',
        'dhonnobad', 'shagotom', 'accha', 'thik', 'hobe', 'hoyeche', 'korecho', 'korechi',
        'tomake', 'amake', 'amader', 'tader', 'oder', 'ekhane', 'okhane', 'kothay', 'kobe',
        'shokal', 'dupur', 'bikel', 'raat', 'brishti', 'rod', 'megh', 'cholo', 'dekha', 'shono',
        'janina', 'jani', 'bujhi', 'bujhina', 'bolchi', 'bolbo', 'bolona', 'khub', 'onek',
        'ektu', 'ekdom', 'kintu', 'tobe', 'tahole', 'jodi', 'tokhon', 'ekhon', 'pore', 'age'
    ]
    
    # Check if message contains Benglish words
    words = user_message.lower().split()
    
    # Check for any Benglish word match (even a single word)
    has_benglish_word = any(word in benglish_indicators for word in words)
    
    # Check for Bengali Unicode characters
    has_bengali_unicode = any(ord(char) >= 0x0980 and ord(char) <= 0x09FF for char in user_message)
    
    # If any Benglish word is found, or any Bengali Unicode character is present, or explicit language preference
    is_bangla = has_benglish_word or has_bengali_unicode or \
                language_preference.lower() in ['bangla', 'bengali']
    
    # Set language instruction based on detection
    language_instruction = "Please respond in Bangla (Bengali) language." if is_bangla else ""
    
    # Store detected language for context
    detected_language = "bangla" if is_bangla else "english"
    
    # Combine vector DB context with conversation history and product context
    vector_context = "\n".join(context_texts) if context_texts else ""
    
    # Build the complete context
    context_parts = []
    
    if conversation_context:
        context_parts.append(f"Previous conversation:\n{conversation_context}")
    
    if product_context:
        context_parts.append(f"Product information:\n{product_context}")
    
    if vector_context:
        context_parts.append(f"Relevant information:\n{vector_context}")
    
    context = "\n\n".join(context_parts) if context_parts else ""
    
    # Check if the database is empty (no text data and no products)
    database_empty = len(text_data) == 0 and len(product_descriptions) == 0
    
    if database_empty:
        # Provide a fallback response when the database is empty
        response = "আমার ডাটাবেস এখনও খালি আছে। দয়া করে অ্যাডমিন প্যানেল থেকে কিছু পণ্যের বিবরণ এবং প্রশিক্ষণ ডাটা যোগ করুন।\n\nMy database is currently empty. Please add some product descriptions and training data from the admin panel."
    else:
        # Get response from Gemini with language instruction and product context
        # Always pass the most_recent_product if it exists, even if not a direct follow-up
        # This ensures the system has context about recently discussed products
        response = get_gemini_response(user_message, context, language_instruction, most_recent_product, product_descriptions)
    
    # Process response to make it more suitable for typewriter effect
    processed_response = response.strip()
    
    # Extract product names from user message
    mentioned_products = extract_product_names(user_message, product_descriptions)
    
    # Extract product names from AI response
    response_products = extract_product_names(processed_response, product_descriptions)
    
    # Combine all mentioned products (using product names for comparison)
    mentioned_product_names = [p['name'] for p in mentioned_products]
    response_product_names = [p['name'] for p in response_products]
    
    # If this is a follow-up and no products were explicitly mentioned, add the most recent product
    if is_followup and most_recent_product and not mentioned_product_names:
        mentioned_products.append({
            'name': most_recent_product,
            'last_mentioned': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        mentioned_product_names.append(most_recent_product)
        
        # Only clear context for follow-up questions, not for product switches
    if is_followup:
        # For Bangla follow-up questions, force the system to focus exclusively on the most recent product
        bangla_indicators = ['ata', 'eta', 'oita', 'eita', 'sheta', 'ei', 'oi', 'kivabe', 'korte', 'hoi', 'hoy', 'korbo', 'use']
        if any(indicator in user_message.lower() for indicator in bangla_indicators):
            # Clear any other products from the context to force focus on the most recent one
            context_texts = []
    
    # Combine all products with their metadata
    all_products = []
    for product in mentioned_products + response_products:
        if product['name'] not in [p['name'] for p in all_products]:
            all_products.append(product)
    
    # Save this exchange to memory with product information
    new_message = {
        'user': user_message,
        'ai': processed_response,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'detected_language': detected_language,
        'mentioned_products': all_products
    }
    
    # Add to messages list
    chat_memory[session_id]['messages'].append(new_message)
    
    # Update session with mentioned products
    if 'recent_products' not in chat_memory[session_id]:
        chat_memory[session_id]['recent_products'] = []
    
    # Update recent products list
    if all_products:
        # Remove existing entries for products that were just mentioned to update them
        product_names = [p['name'] for p in all_products]
        chat_memory[session_id]['recent_products'] = [
            p for p in chat_memory[session_id]['recent_products'] 
            if isinstance(p, dict) and 'name' in p and p['name'] not in product_names
        ]
        
        # Add the updated products to the beginning of the list
        chat_memory[session_id]['recent_products'] = all_products + chat_memory[session_id]['recent_products']
        
        # Keep only the 5 most recent products
        chat_memory[session_id]['recent_products'] = chat_memory[session_id]['recent_products'][:5]
    
    # Save updated chat memory to file
    with open(CHAT_MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(chat_memory, f, indent=4, ensure_ascii=False)
    
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
    # Count total chat sessions and messages
    total_sessions = len(chat_memory)
    total_messages = sum(len(messages) for messages in chat_memory.values())
    
    return render_template('admin_dashboard.html', 
                           text_data=text_data, 
                           product_descriptions=product_descriptions,
                           total_chat_sessions=total_sessions,
                           total_chat_messages=total_messages)

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

@app.route('/admin/add_product', methods=['GET', 'POST'])
@admin_login_required
def add_product():
    """Handle product description addition"""
    global product_descriptions
    
    if request.method == 'POST':
        product_name = request.form.get('product_name', '').strip()
        product_description = request.form.get('product_description', '').strip()
        language = request.form.get('language', 'english').strip()
        
        if not product_name or not product_description:
            flash('Please enter both product name and description.', 'warning')
            return render_template('add_product.html')
        
        # Create new product description entry
        new_product = {
            'id': len(product_descriptions) + 1,
            'name': product_name,
            'description': product_description,
            'language': language,
            'date_added': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add to product descriptions list
        product_descriptions.append(new_product)
        
        # Save to file
        with open(PRODUCT_DESC_FILE, 'w', encoding='utf-8') as f:
            json.dump(product_descriptions, f, indent=4, ensure_ascii=False)
        
        # Also add to vector database for search capability
        combined_text = f"Product: {product_name}\nDescription: {product_description}\nLanguage: {language}"
        add_to_vector_db(combined_text)
        
        flash('Product description added successfully!', 'success')
        return redirect(url_for('product_list'))
    
    return render_template('add_product.html')

@app.route('/admin/products')
@admin_login_required
def product_list():
    """Display list of product descriptions"""
    return render_template('product_list.html', products=product_descriptions)

@app.route('/admin/products/delete/<int:product_id>', methods=['POST'])
@admin_login_required
def delete_product(product_id):
    """Delete a product description"""
    global product_descriptions
    
    # Find and remove the product with the given ID
    product_descriptions = [p for p in product_descriptions if p['id'] != product_id]
    
    # Update IDs to maintain sequence
    for i, product in enumerate(product_descriptions):
        product['id'] = i + 1
    
    # Save updated list to file
    with open(PRODUCT_DESC_FILE, 'w', encoding='utf-8') as f:
        json.dump(product_descriptions, f, indent=4, ensure_ascii=False)
    
    flash('Product description deleted successfully!', 'success')
    return redirect(url_for('product_list'))

@app.route('/admin/chat_history')
@admin_login_required
def view_chat_history():
    """View chat history for all sessions"""
    # Ensure all chat memory entries have the correct structure
    for session_id in chat_memory:
        if not isinstance(chat_memory[session_id], dict):
            chat_memory[session_id] = {
                'messages': chat_memory[session_id] if isinstance(chat_memory[session_id], list) else [],
                'recent_products': []
            }
    return render_template('chat_history.html', chat_memory=chat_memory)

@app.route('/admin/clear_chat_history', methods=['POST'])
@admin_login_required
def clear_chat_history():
    """Clear all chat history"""
    global chat_memory
    
    # Clear chat memory
    chat_memory = {}
    
    # Save empty chat memory to file
    with open(CHAT_MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(chat_memory, f)
    
    flash('Chat history cleared successfully!', 'success')
    return redirect(url_for('view_chat_history'))

@app.route('/admin/clear_database', methods=['POST'])
@admin_login_required
def clear_database():
    """Clear the vector database"""
    global text_data, vectors, vectorizer
    
    # Clear text data and vectors
    text_data = []
    vectors = None
    vectorizer = TfidfVectorizer()
    
    # Save empty data to files
    with open(TEXT_DATA_FILE, 'wb') as f:
        pickle.dump(text_data, f)
    
    with open(VECTORS_FILE, 'wb') as f:
        pickle.dump(vectors, f)
    
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    flash('Vector database cleared successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/clear_products', methods=['POST'])
@admin_login_required
def clear_products():
    """Clear all product descriptions"""
    global product_descriptions
    
    # Clear product descriptions
    product_descriptions = []
    
    # Save empty product descriptions to file
    with open(PRODUCT_DESC_FILE, 'w', encoding='utf-8') as f:
        json.dump(product_descriptions, f, indent=4, ensure_ascii=False)
    
    flash('All product descriptions cleared successfully!', 'success')
    return redirect(url_for('product_list'))

if __name__ == '__main__':
    app.run(debug=True)
