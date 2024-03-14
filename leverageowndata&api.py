import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Get your API Key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.0-pro-latest')

# Initialize sentence embedder
embedder = SentenceTransformer('all-mpnet-base-v2')

# Load and embed data
data_path = "data.txt"

def load_and_embed_data(data_path):
    with open(data_path, 'r') as f:
        data = f.readlines()
        embeddings = embedder.encode(data)
    return data, embeddings

# Build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Process the query
def process_query(query, index, data, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    relevant_docs = ' '.join(data[i] for i in indices[0])

    if distances[0][0] <= 0.5:  # Adjust the threshold as needed
        # Query is related to the data
        context = f"Context: {relevant_docs}\n\nQuery: {query}"
        response = model.generate_content(context)
        return f"Information from data.txt:\n{relevant_docs}\n\nExplanation:\n{response.text}"
    else:
        # Query is not directly related to the data
        context = f"Query: {query}"
        response = model.generate_content(context)
        return response.text

# Load data and build index
data, embeddings = load_and_embed_data(data_path)
index = build_faiss_index(embeddings)

# Streamlit App
def main():
    st.title("Chatbot for Your Data and Gemini")
    user_query = st.text_input("Ask me a question:")

    if user_query:
        response = process_query(user_query, index, data)
        st.write(response)

if __name__ == "__main__":
    main()