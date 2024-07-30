import re
import requests
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def initialize_policy_retriever(url: str, model: str = "text-embedding-3-small"):
    # Fetch the document content
    response = requests.get(url)
    response.raise_for_status()

    # Create a Document object with the fetched content
    doc = Document(page_content=response.text, metadata={"source": url})

    # Initialize the text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

    # Split the document into chunks
    splits = text_splitter.split_documents([doc])
    print(f"Split the document into {len(splits)} chunks.")

    # Initialize the embedding model
    embedding_model = OpenAIEmbeddings(model=model)

    # Create a vector store from the document chunks
    vector_store = Chroma.from_documents(documents=splits, embedding=embedding_model)

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever()

    return retriever


# Define the URL for the document
url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
retriever = initialize_policy_retriever(url)


@tool
def lookup_policy(query: str) -> str:
    """Consult the company policies to check whether certain options are permitted.
    Use this before making any flight changes or performing other 'write' events."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])
