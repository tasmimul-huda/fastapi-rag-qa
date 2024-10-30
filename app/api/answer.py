from fastapi import APIRouter, Request,Body, HTTPException
import logging
from typing import Dict

from rag_pipeline.model_initializer import initialize_models
from rag_pipeline.retriever_chain import RetrieverChain
from settings import Config

answer_router = APIRouter()

logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")
conf = Config()

OPENAI_API_KEY = conf.API_KEY
MODEL_ID = conf.MODEL_ID
MODEL_BASENAME = conf.MODEL_BASENAME
COLLECTION_NAME = conf.COLLECTION_NAME
PERSIST_DIRECTORY = conf.PERSIST_DIRECTORY

print(OPENAI_API_KEY, MODEL_ID, MODEL_BASENAME, PERSIST_DIRECTORY, COLLECTION_NAME)

embedding_model, llm_model = initialize_models(OPENAI_API_KEY,model_id=MODEL_ID, model_basename=MODEL_BASENAME)

def validate_question(data: Dict) -> str:
    """Extract and validate the 'question' field from the incoming data."""
    question = data.get("question")
    if not question or not isinstance(question, str) or not question.strip():
        logger.warning("Received invalid question input.")
        raise HTTPException(status_code=400, detail="Question must be a non-empty string.")
    return question

@answer_router.post('/answer', response_model=dict)
async def generate_answer(data: Dict = Body(...)) -> Dict:
    try:
        # Validate and extract the question
        question = validate_question(data)
        
        # Log incoming question
        logger.info(f"Received question: {question}")
        
        # Generate the answer

        retriever_qa = RetrieverChain(
            collection_name=COLLECTION_NAME, embedding_function=embedding_model, persist_directory=PERSIST_DIRECTORY)
        answer = retriever_qa.get_response(user_input = question, llm= llm_model)
        
        # answer = f"Generated answer for: {question}"
        
        # Log generated answer
        logger.info(f"Generated answer: {answer}")
        
        return {"answer": answer}
    
    except HTTPException as http_exc:
        logger.error(f"HTTP error: {http_exc.detail}")
        raise http_exc  # Re-raise the HTTPException to return the error response
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


