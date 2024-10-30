# Additional Documentation: Retrieval-Augmented Question-Answering System

## Project Objectives
The objective of this project was to develop a robust, question-answering system using the Retrieval-Augmented Generation (RAG) framework. By leveraging a combination of **document retrieval** and **language model** generation, this system was designed to:

1. Accept a question via an API endpoint.
2. Retrieve relevant information from indexed documents.
3. Generate a concise answer using retrieved document segments as context.
4. Handle errors and log critical processes for better observability.
**Design Decisions**
1. **Data Chunking and Format Selection:**
- Decision: Chose a smaller selection of PDF documents and a few compact Excel files for optimal processing.
- Reasoning: Indexing large CSV and Excel datasets proved slow, so the choice of smaller, more efficient data sources improved performance while maintaining relevant information.
- Method: Text is split into chunks maintaining overlape, with splitting adjusted based on document type.
2. **Vector Store Selection:**
- Decision: ChromaDB was selected over FAISS.
- Reasoning: ChromaDB demonstrated better speed and efficiency when working with mixed document types, particularly PDFs and smaller Excel files.
- Implementation: Documents were indexed in ChromaDB for faster similarity search.
3. **Embedding and Model Selection:**
- Decision: Embeddings were generated using HuggingFace’s mpnet-base-v2 model.
- Reasoning: The mpnet-base-v2 model provides high-quality embeddings, suitable for context-rich retrieval tasks.
- Pipeline: When available, OpenAI’s ChatOpenAI model and embeddings are utilized; otherwise, the local Mistral 7B model is used for generation. This setup ensures flexibility and optimizes usage based on available resources.
4. **RAG Pipeline with LangChain:**
- Decision: LangChain was chosen for seamless integration of retrieval and generation steps.
- Reasoning: LangChain’s flexible API supports chaining retrieval and language model components, making it a natural fit for the RAG pipeline requirements.
## Error Handling and Logging
**Error Handling**
1. **Invalid Input Types:**
- Checks if the question field is provided as a string.
- Returns HTTP status 422 with an error message if the input is invalid.
2. **Malformed JSON:**
- Detects improperly formatted JSON requests.
- Responds with an HTTP 400 status and a relevant error message.
3.**Retrieval or Generation Errors:**
- Any errors during document retrieval or generation are caught and handled, with meaningful error messages returned to the client.
**Logging**
1. **Stream and File Handlers:**
- Logging Setup: Logging utilizes both a StreamHandler for real-time output and a FileHandler to save logs.
- File Handling: Log files are saved with a maximum size of 5MB and a limit of 2 backup files.
2. **Events Logged:**
- Incoming Requests: Each request is logged with details like question content and timestamp.
- Critical Errors: Logs error details and stack traces for debugging.
- Retrieval and Generation Events: Events such as document retrieval counts, model used, and response time are logged for tracking and analysis.

## Testing Details
The project includes a suite of unit tests covering key components of the RAG pipeline.

**Retrieval Function Tests**
1. ***Valid Retrieval Test:***
-Ensures the retriever fetches relevant document chunks based on a valid question.
2. ***Invalid Retrieval Query Test:***
-Tests retrieval on an invalid or nonsensical query to ensure it does not fail unexpectedly.

**API Endpoint Tests**
1. ***test_valid_question:***
- Verifies that a valid question returns an HTTP 200 response with a generated answer.
2. ***test_missing_question_field:***
- Ensures that requests missing the question field return an HTTP 422 status.
3. ***test_invalid_question_type:***
- Checks that incorrect data types (e.g., integer or list instead of a string for question) are properly handled with a 422 status code.

4. ***test_malformed_json:***
- Confirms that malformed JSON in the request results in an HTTP 400 response.
## Tools Used
* Testing Framework: pytest for testing framework consistency.
* Mocking: Mocks were used to simulate retrieval and generation to focus on integration logic.

## Assumptions and Trade-offs
1. **Assumptions:**
* Documents provided will contain enough information to adequately answer questions within the chosen domains.
* Either OpenAI API keys or local model resources (like Mistral 7B) are available for generating responses, depending on the environment.
* LangChain compatibility is maintained for both retrieval and generation.

2. **Trade-offs:**
* Vector Storage: ChromaDB was chosen over FAISS due to its faster performance with mixed document types (PDFs, DOcx.smaller Excel files).
* Embedding Model Choice: The mpnet-base-v2 embedding model strikes a balance between relevance and computational efficiency.
* Local vs. API-based Model: Mistral 7B is used as a local model option, but switching to OpenAI models improves accuracy when available, adding flexibility at the expense of dependency on external API keys.
3. **Limitations:**
* Resource Demands: Running larger models locally (like Mistral 13B) can be computationally intensive.