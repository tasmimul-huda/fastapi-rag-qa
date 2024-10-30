from fastapi import FastAPI, APIRouter, UploadFile, HTTPException
from fastapi import File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import List
import os
import shutil
import logging


from app.data_pipeline.data_loader import DocumentLoader
from app.data_pipeline.embedding_manager import split_text,initialize_embedding_model,create_and_store_embeddings
import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

from app.settings import Config
conf = Config()

upload_router = APIRouter()

UPLOAD_DIR = conf.UPLOAD_DIR

COLLECTION_NAME = conf.COLLECTION_NAME
PERSIST_DIRECTORY = conf.PERSIST_DIRECTORY


# Type of files allowed to be uploaded
def is_allowed_file(filename):
    allowed_extensions = {"pdf", "csv", "doc", "docx", "txt", "xlsx", "xls"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def empty_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate through all items in the folder
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            # Remove files and folders
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"The folder '{folder_path}' has been emptied.")
    else:
        print(f"The folder '{folder_path}' does not exist.")




@upload_router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        # Empty the upload directory
        empty_folder(UPLOAD_DIR)
        logger.info(f"{UPLOAD_DIR} is now empty.")

        # Check if UPLOAD_DIR exists
        if not os.path.exists(UPLOAD_DIR):
            logger.error(f"Upload directory '{UPLOAD_DIR}' does not exist.")
            return JSONResponse(content={"error": f"Folder '{UPLOAD_DIR}' does not exist"}, status_code=404)

        # Save uploaded files
        for uploaded_file in files:
            if not is_allowed_file(uploaded_file.filename):
                logger.error(f"File type of '{uploaded_file.filename}' not allowed.")
                return JSONResponse(content={"error": "File type not allowed"}, status_code=400)

            file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(uploaded_file.file.read())
            logger.info(f"File '{uploaded_file.filename}' uploaded successfully.")

        # Load documents from the upload directory
        try:
            document_loader = DocumentLoader(UPLOAD_DIR)
            documents = document_loader.load_all_documents()
            logger.info(f"Loaded {len(documents)} documents.")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return JSONResponse(content={"error": "Failed to load documents"}, status_code=500)

        # Process documents into chunks for embedding
        try:
            chunks = split_text(documents)
            logger.info(f"Processed {len(chunks)} chunks for embedding.")
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return JSONResponse(content={"error": "Failed to process documents"}, status_code=500)

        # Initialize the embedding model
        try:
            embedding_function = initialize_embedding_model()
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            return JSONResponse(content={"error": "Failed to initialize embedding model"}, status_code=500)

        # Create and store embeddings
        try:
            create_and_store_embeddings(chunks, COLLECTION_NAME, embedding_function, PERSIST_DIRECTORY)
            logger.info("Embeddings created and stored successfully.")
        except Exception as e:
            logger.error(f"Error creating or storing embeddings: {e}")
            return JSONResponse(content={"error": "Failed to create and store embeddings"}, status_code=500)

        # Return success message if everything is successful
        return JSONResponse(content={"message": "Documents successfully loaded and processed."})

    except Exception as e:
        logger.error(f"Unexpected error in upload_files endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")














# @upload_router.post("/upload")
# async def upload_files(files: List[UploadFile] = File(...)):

#     empty_folder(UPLOAD_DIR)
#     logger.info(f" {UPLOAD_DIR} is empty Now")

#     if not os.path.exists(UPLOAD_DIR):
#         logger.error(f"{UPLOAD_DIR}' does not exist")
#         return JSONResponse(content={"error": f"Folder '{UPLOAD_DIR}' does not exist"}, status_code=404)

#     for uploaded_file in files:
#         if not is_allowed_file(uploaded_file.filename):
#             logger.error(f"File type not allowed")
#             return JSONResponse(content={"error": "File type not allowed"}, status_code=400)

#         file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
#         with open(file_path, "wb") as buffer:
#             buffer.write(uploaded_file.file.read())

#         logger.info(f"Files uploaded successfully")
    
#     try: 
#         document_loader = DocumentLoader(UPLOAD_DIR)
#         documents = document_loader.load_all_documents()
#         logger.info(f"Loaded {len(documents)} documents.")
#     except Exception as e:
#         logger.error(f"Error loading documents: {e}")
#         return

#     try:
#         chunks = split_text(documents)
#         logger.info(f"Processed {len(chunks)} chunks for embedding.", )
#     except Exception as e:
#         logger.error(f"Error processing documents: {e}")
#         return
    
#     try:
#         embedding_function = initialize_embedding_model()
#     except Exception:
#         return  # Stop execution if embedding model fails

#     create_and_store_embeddings(chunks, COLLECTION_NAME, embedding_function, PERSIST_DIRECTORY)
#     logger.info(f'Documents Successfully loades')
#     return JSONResponse(content={"message": "Documents Successfully loades"})
