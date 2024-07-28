import os
from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure the Google API key
genai.configure(api_key="AIzaSyAR7cD2I6uyC_hKnmNYYxHe2gtwlc1nX4o")  # Replace with your actual API key

# Function to read and process PDFs from the specified directory
def read_pdfs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            loader = PyMuPDFLoader(os.path.join(directory, filename))
            # Load the PDF document
            pdf_documents = loader.load()
            for doc in pdf_documents:
                # Append document and prompt to list
                documents.append({"document": doc})

    return documents

# Function to split text into chunks using RecursiveCharacterTextSplitter
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents([doc["document"] for doc in documents])
    return texts

# Function to embed documents and create a FAISS vector store
def create_vector_store(documents):
    texts = split_text(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyAR7cD2I6uyC_hKnmNYYxHe2gtwlc1nX4o")  # Replace with your actual API key
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store, documents

# Load and embed documents from the specified directory
pdf_directory = 'C:/Users/tejas/Downloads/Disseration_train_Gemini_AI/incident_pdfs'
documents = read_pdfs(pdf_directory)
vector_store, indexed_documents = create_vector_store(documents)

# Function to setup the Conversational Chain with Google Generative AI
def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key="AIzaSyAR7cD2I6uyC_hKnmNYYxHe2gtwlc1nX4o")  # Replace with your actual API key
    prompt_template = """
    You are an AI assistant that helps answer questions based on the provided context. Use the context to answer the question as accurately as possible.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return qa_chain

qa_chain = get_conversational_chain()

# Function to take user input and get a response from the chain
def get_response(user_input, vector_store, qa_chain):
    docs = vector_store.similarity_search(user_input)
    if not docs:
        return "No relevant documents found."
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Logging the context and question for debugging purposes
    print(f"Context provided to the model:\n{context}")
    print(f"Question: {user_input}")
    
    question_input = {"context": context, "question": user_input, "input_documents": docs}
    response = qa_chain(question_input, return_only_outputs=True)
    
    # Logging the response for debugging purposes
    print(f"Response from qa_chain:\n{response['output_text']}")
    
    return response['output_text']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip()
    if not user_input:
        return jsonify({"error": "Input cannot be empty"}), 400
    
    response_text = get_response(user_input, vector_store, qa_chain)
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)

