# LLM Chat Application with Gemini 1.5 Pro and Vector Database

This application is a Flask-based chatbot that uses Google's Gemini 1.5 Pro model API for generating responses. It stores and retrieves information using a TF-IDF vector database, allowing for semantic search and context-aware conversations.

## Features

- Flask web application with a chat interface
- Integration with Google's Gemini Flash 2.0 model API
- FAISS vector database for efficient similarity search
- Daily data ingestion capability
- Contextual conversation based on stored knowledge

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```
   python app.py
   ```

## Usage

- Access the web interface at `http://localhost:5000`
- Use the chat interface to interact with the LLM
- Add new data through the data ingestion page
