import os
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from rag_pipeline.model_loader import load_model

logger = logging.getLogger(__name__)

def initialize_models(openai_api_key=None,model_id=None, model_basename=None):
    """
    Initializes embedding and chat model based on the OpenAI API key availability.
    
    Returns:
        tuple: (embedding_model, llm_model)
    """
    
    try:
        if openai_api_key:
            embedding_model = OpenAIEmbeddings(api_key=openai_api_key)
            llm_model = ChatOpenAI(api_key=openai_api_key)
            logger.info("Using OpenAI models.")
        else:
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", 
                                                    model_kwargs={'device': 'cpu'}, 
                                                    encode_kwargs={'normalize_embeddings': False})
            llm_model = load_model(device_type="cpu", model_id=model_id, model_basename=model_basename)
            logger.info("Using Hugging Face embeddings and local LLM model.")
        
        return embedding_model, llm_model

    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise
