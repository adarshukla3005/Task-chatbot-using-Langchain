from flask import Flask, render_template, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
from langchain_community.embeddings import SentenceTransformerEmbeddings
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise ValueError("HF_API_TOKEN is missing. Please set it in the .env file or as an environment variable.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Extract Data from the Website
def extract_data_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure successful request
        soup = BeautifulSoup(response.text, "html.parser")
        
        content = soup.get_text(separator="\n")  # Extract page text
        documents = [Document(page_content=content, metadata={"source": url})]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error extracting data from URL: {e}")
        return []

# Create and Save Vector Store
def create_and_save_vector_store(documents, vector_store_path="vector_store"):
    try:
        model_name = 'all-MiniLM-L6-v2'
        embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        texts = [doc.page_content for doc in documents]
        
        vector_store = FAISS.from_texts(texts=texts, embedding=embeddings)
        vector_store.save_local(vector_store_path)
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

# Load the Vector Store
def load_vector_store(vector_store_path="vector_store"):
    try:
        model_name = 'all-MiniLM-L6-v2'
        embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        vector_store = FAISS.load_local(vector_store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

# Define routes
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if request.method == "POST":
            if request.content_type != 'application/json':
                return jsonify({"response": "Unsupported Media Type"}), 415

            user_input = request.json.get("message")
            if not user_input:
                return jsonify({"response": "Please provide a message!"}), 400

            response = qa_chain.run(user_input)
            return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error: {e}"}), 500

if __name__ == "__main__":
    try:
        url = "https://brainlox.com/courses/category/technical"
        print("Extracting data from the URL...")
        docs = extract_data_from_url(url)
        
        if not docs:
            print("No documents extracted. Exiting...")
            exit(1)
        
        print("Creating embeddings and vector store...")
        vector_store_path = "vector_store"
        vector_store = create_and_save_vector_store(docs, vector_store_path)
        
        if vector_store is None:
            print("Vector store creation failed. Exiting...")
            exit(1)
        
        print("Loading the vector store for retrieval...")
        vector_store = load_vector_store(vector_store_path)
        
        if vector_store is None:
            print("Vector store loading failed. Exiting...")
            exit(1)
        
        retriever = vector_store.as_retriever()
        
        # Set up the QA chain
        print("Setting up the QA chain...")
        question_answering_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")
        llm = HuggingFacePipeline(pipeline=question_answering_pipeline)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        
        print("Starting Flask server...")
        app.run(port=5000)
    except Exception as e:
        print(f"Error in server setup: {e}")