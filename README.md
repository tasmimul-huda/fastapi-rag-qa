# AI Assignment: Retrieval-Augmented Question-Answering System with FastAPI and LangChain

## **Overview**
This project implements a question-answering service that utilizes Retrieval-Augmented Generation (RAG). By leveraging FastAPI and LangChain, the application answers user questions based on a pre-indexed set of documents. The RAG pipeline first retrieves relevant information from indexed document chunks and then generates an answer using a language model conditioned on the retrieved data. The design prioritizes efficiency, error handling, and modularity.

## Project Setup

1. **Clone the Repository**  
   Extract the project zip or clone the repository.

2. **Create a Virtual Environment**  
   Create a Virtual Environment using Python 3.10:  
   ```bash
   virtualenv -p python3.10 env
   ```

3. **Activate Virtual Environment**  
   Activate the virtual environment:
* On Windows:
```bash
env\Scripts\activate
```
* On macOS/Linux:
```bash
source env/bin/activate
```

4. **Install Dependencies**
```bash
pip install -r requirements.txt
```
5. **Set Up Environment Variables**  
   Open the .env file and add your `OpenAI API` key if you have any. By default it will run with `mistral-7b-v0.1.Q4_0.gguf` model
   ```bash
   OPENAI_API_KEY="your_openai_api_key_here"
   ```

## **Data Preparation & Embedding**
- The Data folder contains all the documents needed for processing. Or You should all documents in this folder
- To create embeddings and store them in the Chroma `vector database`, navigate to the `data_pipeline` folder and run the `embedding_manager.py` script:
```bash
cd data_pipeline
python embedding_manager.py
```
- This script will create a `vector_store` folder, storing vectorized data for retrieval.

## Running the FastAPI Application
1. Navigate to the `application` Directory:
  ```bash
  cd application
  ```
2. Start the FastAPI server:
```bash
uvicorn main:app
```
or
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
3. Accessing the API:
* Request Format: Ensure to send requests in the following format:
```bash
{"question": "Your question here"}
```
* Response Format: Ensure to send requests in the following format:
```bash
{"answer": "Generated answer here"}
```

## Running Tests:
- The test suite includes validations for both retrieval and question-answering functionalities.
1. Navigate to the Tests Folder:
```bash
cd tests
```
2. Run Tests:
```bash
 pytest test_answer.py
```
```bash 
pytest test_retriever_chain.py
```
- Alternatively, run all tests with:
```bash
pytest
```

## Basic Architecture Summary

* **Data Preparation**: Preprocesses documents, removing unnecessary characters and splitting text into manageable chunks.
* Vector Indexing: Stores embeddings in a vector database (ChromaDB).
* Retrieval-Augmented Generation (RAG) Pipeline:\
  i. Retrieval: Finds relevant document chunks based on the user's question.\
  ii. Generation: Utilizes a language model (OpenAI GPT-3.5 or equivalent local model) to generate answers conditioned on the retrieved documents.

## Error Handling and Logging
**Error Handling:**
* Returns 422 for invalid input types.
* Returns 400 for malformed JSON.
* Manages missing fields and unexpected retrieval errors gracefully.
**Logging:**
* Logs incoming requests, critical errors, and retrieval/generation process events.
  
**Design Decisions**
***Document Chunking:*** Improves retrieval efficiency by splitting long documents into chunks.
***Vector Store Selection:*** ChromaDB used for indexing to optimize similarity search. I choose ChromaDb over FAISS, Because in my case it took longer to store embedding in faiss than chroma
***LangChain for Data Loadeing & Model Integration:*** Provides a seamless interface to load various types of data and manage retrieval and language generation.

**Testing Strategy**
***test_valid_question:*** Ensures the endpoint returns a 200 status and an answer when given a valid question.
***test_missing_question_field:*** Verifies a 422 error is returned for requests missing the question field.
***test_invalid_question_type:*** Ensures a 422 error if the question field is of an invalid type.
|***test_malformed_json:*** Confirms a 400 error for improperly formatted JSON inputs.

**Assumptions, Trade-offs, and Limitations**
Assumptions:
The provided dataset is sufficient for answering most questions.
OpenAI’s GPT-3.5 (or similar model) is available for generation.

Trade-offs:
Using a pretrained language model limits the control over fine-tuning specific domain knowledge.

**Limitations:**
* System’s accuracy depends on model and document relevancy.
* Larger documents may affect retrieval speed due to vector indexing constraints.