import os
import logging
from typing import List
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


from settings import Config

conf = Config()

OPENAI_API_KEY = conf.API_KEY
PERSIST_DIRECTORY = conf.PERSIST_DIRECTORY
COLLECTION_NAME = conf.COLLECTION_NAME


# Set up logging
import logging

logger = logging.getLogger(__name__)

def initialize_embedding_model():
    """Initialize the embedding model based on the availability of the OpenAI API key."""
    try:
        if OPENAI_API_KEY:
            logger.info("Using OpenAI embedding model.")
            embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        else:
            logger.info(f"Using Hugging Face embedding model.")
            embedding_model = HuggingFaceEmbeddings(
                model_name=conf.MODEL_NAME,
                model_kwargs=conf.MODEL_KWARGS,
                encode_kwargs=conf.ENCODE_KWARGS
            )
        return embedding_model
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        raise



def split_text(documents: List[str]) -> List[str]:
    """Split documents into smaller chunks."""
    try:
        logger.info(f"Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=conf.CHUNK_SIZE, chunk_overlap=conf.CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Document splitting completed.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        raise

def get_chroma_client(collection_name: str, embedding_function, persist_directory: str):
    """Initialize and return a Chroma client for a specific collection."""
    try:
        logger.info(f"Creating Chroma client for collection: {collection_name}")
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
    except Exception as e:
        logger.error(f"Error creating Chroma client:  {e}")
        raise

def create_and_store_embeddings(chunks: List[str], collection_name: str, embedding_function, persist_directory: str):
    """Create and store embeddings for document chunks."""
    try:
        vector_db = get_chroma_client(collection_name, embedding_function, persist_directory)
        vector_db.add_documents(chunks)
        logger.info(f"Embeddings created for collection {collection_name} and saved to {persist_directory}.")
    except Exception as e:
        logger.error("Error creating and storing embeddings: {e}")
        raise

# def main():
#     source_directory = conf.DATA_DIRECTORY
#     document_loader = DocumentLoader(source_directory)
#     try:
#         documents = document_loader.load_all_documents()
#         logger.info(f"Loaded {len(documents)} documents.")
#     except Exception as e:
#         logger.error(f"Error loading documents: {e}")
#         return

#     # Split documents into chunks
#     try:
#         chunks = split_text(documents)
#         logger.info(f"Processed {len(chunks)} chunks for embedding.", )
#     except Exception as e:
#         logger.error(f"Error processing documents: {e}")
#         return

#     # Initialize embedding model
#     try:
#         embedding_function = initialize_embedding_model()
#     except Exception:
#         return  # Stop execution if embedding model fails

#     # Create and store embeddings
#     create_and_store_embeddings(chunks, COLLECTION_NAME, embedding_function, PERSIST_DIRECTORY)

# if __name__ == "__main__":
#     main()


