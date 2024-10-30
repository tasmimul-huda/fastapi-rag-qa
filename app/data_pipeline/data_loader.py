import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.docstore.document import Document

import logging

INGEST_THREADS = os.cpu_count() or 8

DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".csv": UnstructuredCSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader
    # Add additional file types here if necessary
}

logger = logging.getLogger(__name__)

class DocumentLoader():
    def __init__(self, source_dir: str):
        """
        Initializes the loader with the directory path from which to load documents.
        """
        self.source_dir = source_dir
        logger.info(f"DocumentLoader initialized with source directory: {self.source_dir}")

    def load_single_document(self, file_path: str):
        """
        Loads a single document based on its file extension using the appropriate loader.

        Args:
            file_path (str): Path to the document file.

        Returns:
            List[Document]: Loaded document(s) as LangChain Document instances.
        """

        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)

        if loader_class: 
            loader = loader_class(file_path)
            logger.info(f"Loading document: {file_path}")
            try:
                documents = loader.load()
                logger.info(f"Successfully loaded document: {file_path}")
                return documents
            except Exception as e:
                logger.error(f"Error loading document {file_path}: {e}", exc_info=True)
                raise
        else:
            logger.warning(f"Unsupported document type for file: {file_path}")
            raise ValueError(f"Unsupported document type: {file_extension}")
        

    def load_all_documents(self) -> list[Document]:
        """
        Loads all documents from the source directory, including documents in subdirectories.

        Returns:
            List[Document]: List of all loaded documents from the source directory.
        """
        paths = self._gather_file_paths()  # Gather file paths of documents to load
        all_docs = []

        logger.info(f"Loading all documents from directory: {self.source_dir}")

        # # Load each document sequentially
        # for file_path in paths:
        #     documents = self.load_single_document(file_path)
        #     all_docs.extend(documents)  # Append loaded documents to the result list

        # # return all_docs
    

        for file_path in paths:
            try:
                documents = self.load_single_document(file_path)
                all_docs.extend(documents)  # Append loaded documents to the result list
            except ValueError as e:
                logger.error(f"Skipping file {file_path}: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while loading {file_path}: {e}", exc_info=True)

        logger.info(f"Finished loading documents. Total documents loaded: {len(all_docs)}")
        return all_docs

    def _gather_file_paths(self):
        """
        Walks through the source directory and gathers file paths of documents 
        that match the supported file types in DOCUMENT_MAP.

        Returns:
            List[str]: List of file paths for documents to load.
        """
        file_paths = []
        logger.debug(f"Scanning for files in directory: {self.source_dir}")
        for root, _, files in os.walk(self.source_dir):
            for file_name in files:
                file_extension = os.path.splitext(file_name)[1]
                if file_extension in DOCUMENT_MAP:
                    full_path = os.path.join(root, file_name)
                    file_paths.append(full_path)
                    logger.debug(f"Found document: {full_path}")
                    
        logger.info(f"Total files found for loading: {len(file_paths)}")
        return file_paths



if __name__ == "__main__":
    source_directory = os.path.join(os.path.dirname(__file__),'..','Data')
    document_loader = DocumentLoader(source_directory)
    
    documents = document_loader.load_all_documents()






















# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# directory_path = os.path.join(os.path.dirname(__file__),'..','Data')
# documents = load_documents(directory_path)
# print(documents)

# print(os.path.join(os.path.dirname(__file__),'..','Data'))