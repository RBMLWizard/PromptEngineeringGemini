#https://github.com/RBMLWizard/PromptEngineeringGemini
#This is a personal project where i am learning capabilities for various use cases.

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
embedder = SentenceTransformer('all-MiniLM-L6-v2')  

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

# Query processing and response generation
def process_query(query, index, data, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)

    if np.any(distances[0] < 0.5):  
        relevant_docs = ' '.join(data[i] for i in indices[0])
        context = f"Context: {relevant_docs}\n\nQuery: {query}"
        response = model.generate_content(context)  
        return response.text

    else:  
        # Directly query Gemini 
        # *** Adjust this based on documentation: ***
        response = model.generate_content(query)  # Or the correct method name
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
