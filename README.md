# Task-chatbot-using-Langchain

![AI_Chatbot](https://github.com/user-attachments/assets/bebaef03-b707-490a-9940-f45472b472a5)


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
   git clone https://github.com/adarshukla3005/Task-chatbot-using-Langchain.git
   cd Task-chatbot-using-Langchain
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Set up your .env file with your HF_API_TOKEN:
   ```bash
   HF_API_TOKEN=your_huggingface_token_here

4. Run the Flask server:
   ```bash
   python app.py

5. Open the app in your browser at http://localhost:5000
