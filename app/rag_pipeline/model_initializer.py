import os
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from app.rag_pipeline.model_loader import load_model
from langchain_huggingface import HuggingFacePipeline
from app.settings import Config
conf = Config()

CACHE_DIR = conf.CACHE_DIR

logger = logging.getLogger(__name__)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

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
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", # all-mpnet-base-v2
                                                    model_kwargs={'device': 'cpu'}, 
                                                    encode_kwargs={'normalize_embeddings': False},
                                                    cache_folder = CACHE_DIR
                                                    )
            # llm_model = load_model(device_type="cpu", model_id=model_id, model_basename=model_basename, LOGGING=logger)
            llm_model = HuggingFacePipeline.from_model_id(
                model_id= "gpt2", #"google/flan-t5-small",
                task="text-generation",
            )

            #TheBloke/Mistral-7B-v0.1-GGUF
            #HuggingFaceH4/zephyr-7b-beta
            
            logger.info("Using Hugging Face embeddings and local LLM model.")
        
        return embedding_model, llm_model

    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise
