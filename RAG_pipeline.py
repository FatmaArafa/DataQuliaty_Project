import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import openai
import time
from openai import OpenAIError, RateLimitError

def call_openai_with_retry(prompt, max_retries=3):
    """Calls OpenAI API with retry logic for handling rate limits."""
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            return response
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
        except OpenAIError as e:
            return f"An error occurred: {e}"

def get_response_from_openai(prompt):
    """Handles interaction with OpenAI using retry logic."""
    try:
        response = call_openai_with_retry(prompt)
        return response["choices"][0]["message"]["content"]
    except RateLimitError:
        return "Request failed due to rate limits. Please try again later."
    except OpenAIError as e:
        return f"An unexpected error occurred: {e}"


def preprocess_data(filepath):
    """Loads and processes the dataset for RAG."""
    df = pd.read_csv(filepath)  # Adjust based on your file type
    documents = df.apply(lambda row: " ".join(map(str, row)), axis=1).tolist()
    return documents

def create_vector_store(documents):
    """Creates a vector store from documents."""
    embeddings = OpenAIEmbeddings()  # Requires OpenAI API key
    vector_store = FAISS.from_texts(documents, embedding=embeddings)
    return vector_store

def setup_rag_system(vector_store):
    """Sets up the RAG system."""
    retriever = vector_store.as_retriever(search_type="similarity", search_k=5)
    generator = ChatOpenAI(model="gpt-4")  # Or "gpt-3.5-turbo"
    qa_chain = RetrievalQA.from_chain_type(llm=generator, retriever=retriever)
    return qa_chain
