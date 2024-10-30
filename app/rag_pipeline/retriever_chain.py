import logging
#import create_history_aware_retriever, 
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from rag_pipeline.prompt_utils import qa_prompt
from rag_pipeline.chroma_client import get_chroma_client
from settings import Config

# from prompt_utils import qa_prompt
# from chroma_client import get_chroma_client




# import sys
# import os

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, parent_dir)
# from settings import Config


conf = Config()


MODELS_PATH =  conf.MODELS_PATH #'/models'

CONTEXT_WINDOW_SIZE = 2048
MAX_NEW_TOKENS = 2048
N_BATCH= 512
N_GPU_LAYERS = 1

MODEL_ID = conf.MODEL_ID  #"TheBloke/Mistral-7B-v0.1-GGUF"
MODEL_BASENAME = conf.MODEL_BASENAME # "mistral-7b-v0.1.Q4_0.gguf"
device_type = 'cpu'

logger = logging.getLogger(__name__)

class RetrieverChain:
    def __init__(self, collection_name, embedding_function, persist_directory):
        try:
            self.vector_db = get_chroma_client(collection_name, embedding_function, persist_directory)
        except Exception as e:
            logger.error(f"Error creating RetrieverChain: {e}")
            raise

    def get_retriever(self):
        try:
            retriever = self.vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 2})
            
            return retriever
        except Exception as e:
            logger.error(f"Failed to get retriever: {e}")
            raise

    def get_conversational_rag_chain(self, llm):
        try:

            if self.get_retriever is None:
                logger.error(f"Retriever must not be None")
                raise ValueError("Retriever must not be None")
            if llm is None:
                logger.error(f"Model must not be None")
                raise ValueError("Model must not be None")

            
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            return create_retrieval_chain(self.get_retriever(), question_answer_chain)
        except Exception as e:
            logger.error(f"Error creating RAG chain: {e}")
            raise

    
    def get_relevent_docs(self, user_input):

        try:
            docs = self.vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 3}).get_relevant_documents(user_input)
            logger.info(f"Relevent documents for {user_input}: {docs}")
            # Access the retrieved documents

            # print("Relevent Docs")
            # for doc in docs:
            #     print(doc.page_content)  # Access the original text
            #     print(doc.metadata)  # Access any metadata associated with the document
            # print("Relevent Docs end")
            return docs

        except Exception as e:
            logger.error(f"Error getting response: {e}")
            raise

    def get_response(self, user_input, llm):
        try:   
            qa_rag_chain = self.get_conversational_rag_chain(llm)
            response = qa_rag_chain.invoke({"input": user_input})
            return response['answer']
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            raise






class RetrieverChainNew:
    def __init__(self, collection_name, embedding_function, persist_directory):
        try:
            self.vector_db = get_chroma_client(collection_name, embedding_function, persist_directory)
        except Exception as e:
            logger.error(f"Error creating RetrieverChain: {e}")
            raise

    def get_retriever(self):
        try:
            return self.vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 2})
        except Exception as e:
            logger.error(f"Failed to get retriever: {e}")
            raise

    def retrieve_documents(self, user_input):
        """
        Retrieve relevant documents based on user query.
        """
        try:
            retriever = self.get_retriever()
            docs = retriever.get_relevant_documents(user_input)
            logger.info(f"Retrieved {len(docs)} documents for query: {user_input}")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    def generate_response(self, docs, llm):
        """
        Generate a response from the language model based on retrieved documents.
        """
        try:
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            # Passing the retrieved documents to the QA chain for answer generation
            response = question_answer_chain.invoke({"documents": docs})
            return response['answer']
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def get_response(self, user_input, llm):
        """
        Retrieve documents and generate a response based on user query.
        """
        try:
            # Step 1: Retrieve relevant documents
            docs = self.retrieve_documents(user_input)
            
            # Step 2: Generate a response using retrieved documents
            answer = self.generate_response(docs, llm)
            
            return answer
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            raise





if __name__ == "__main__":
    import os
    from model_initializer import initialize_models
    
    
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    openai_api_key = conf.API_KEY

    embedding_model, llm_model = initialize_models(openai_api_key,model_id=MODEL_ID, model_basename=MODEL_BASENAME)

    print(f"embeddi_modelng: {embedding_model}")
    print(f"llm_model: {llm_model}")

    collection_name = 'AI_assignment'

    persist_directory = f'D:/AI Assignment/vector_store'
    print(f"persist_directory: {persist_directory}")
    while True:
        print("Enter query: ")
        user_query = input()
        if user_query.lower() == 'exit':
            break

        retriever_qa = RetrieverChain(
            collection_name=collection_name, embedding_function=embedding_model, persist_directory=persist_directory)
        response = retriever_qa.get_response(user_input = user_query, llm= llm_model)
        print(f"Response: {response}")





