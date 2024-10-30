from langchain_chroma import Chroma
import logging 


logger = logging.getLogger(__name__)

def get_chroma_client(collection_name, embedding_function, persist_directory):
    try:
        return Chroma(collection_name=collection_name,
                      embedding_function=embedding_function,
                      persist_directory=persist_directory)
    except Exception as e:
        logging.error(f"Failed to initialize Chroma client: {e}")
        raise