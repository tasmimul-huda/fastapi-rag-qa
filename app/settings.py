import os
from dotenv import load_dotenv

from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

class Config:
    API_KEY = os.getenv('OPENAI_API_KEY')
    MODEL_ID = os.getenv('MODEL_ID')
    MODEL_BASENAME = os.getenv('MODEL_BASENAME')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME')

    PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__),'..','vector_store')
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    UPLOAD_DIR = os.path.join(os.path.dirname(__file__),'..','uploads')
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    LOG_DIR = os.path.join(os.path.dirname(__file__),'..','log_dir')
    os.makedirs(LOG_DIR, exist_ok=True)

    MODELS_PATH = os.path.join(os.path.dirname(__file__),'..','models')
    os.makedirs(MODELS_PATH, exist_ok=True)
    # MODELS_PATH = '/models'

    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    MODEL_KWARGS = {'device': 'cpu'}
    ENCODE_KWARGS = {'normalize_embeddings': False}
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200