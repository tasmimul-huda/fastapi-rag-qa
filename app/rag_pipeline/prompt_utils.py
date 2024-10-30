from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


contex_retriever_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])


conversion_retriever_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])


system_prompt = (
    "You are an assistant specializing in answering questions accurately based on provided context. "
    "Use the context to answer the question concisely. If the answer is not found in the context, respond with 'I'm not sure'."
    "\n\n"
    "Context:\n{context}\n\n"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)
