# Task-chatbot-using-Langchain

# Brainlox Chatbot

A smart chatbot that answers questions related to courses offered on Brainlox. Built with Flask, Langchain, Hugging Face, and powerful NLP models, this web application allows users to interact with extracted data from Brainlox's technical courses.

## 🚀 Features

- **Ask Questions**: Users can ask anything related to Brainlox courses, and the chatbot responds using an advanced NLP-powered backend.
- **Real-Time Response**: Receive instant replies to your queries with minimal latency, thanks to Flask and Hugging Face models.
- **Dynamic Data Extraction**: Extracts course data directly from the Brainlox website for up-to-date responses.
- **Sleek UI**: Simple, intuitive, and responsive design for an engaging user experience.

## 🛠️ Technologies Used

- **Flask**: Lightweight backend server for handling API requests.
- **Langchain**: For integrating language models and creating a robust question-answering pipeline.
- **Hugging Face**: Utilized for pre-trained language models like BART for text generation.
- **Sentence Transformers**: Used for generating embeddings and creating a vector store for document retrieval.
- **FAISS**: Efficient similarity search for fast document retrieval.

## ⚡ Getting Started

### Prerequisites

Make sure you have Python 3.x installed, along with these packages:

- Flask
- Langchain
- Hugging Face
- Sentence Transformers
- FAISS
- Requests
- BeautifulSoup
- dotenv
- flask_cors

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brainlox-chatbot.git
   cd brainlox-chatbot
