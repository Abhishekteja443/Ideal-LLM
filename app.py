import streamlit as st
import chromadb
import ollama
import json
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

# Initialize ChromaDB client
embedding_function = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text:latest",
)
client = chromadb.PersistentClient(path="./demo-rag-chroma")

def get_collection():
    """
    Retrieve the ChromaDB collection where documents are stored.
    """
    return client.get_or_create_collection(
        name="site_pages_collection",
        embedding_function=embedding_function
    )

def get_relevant_chunks(query: str, top_k: int = 5):
    """
    Retrieve the top K relevant chunks from the vector database for a given query.
    """
    collection = get_collection()
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        return documents, metadatas
    except Exception as e:
        st.error(f"Error querying the database: {e}")
        return [], []

def ask_llm(query: str, retrieved_chunks: list):
    """
    Send the user query and retrieved chunks to the LLM and get a response.
    """
    system_prompt = """
    You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

    context will be passed as "Context:"
    user question will be passed as "Question:"

    To answer the question:
    1. Thoroughly analyze the context, identifying key information relevant to the question.
    2. Organize your thoughts and plan your response to ensure a logical flow of information.
    3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
    4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
    5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

    Format your response as follows:
    1. Use clear, concise language.
    2. Organize your answer into paragraphs for readability.
    3. Use bullet points or numbered lists where appropriate to break down complex information.
    4. If relevant, include any headings or subheadings to structure your response.
    5. Ensure proper grammar, punctuation, and spelling throughout your answer.

    Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
    """

    context = "\n\n".join(retrieved_chunks)
    llm_prompt = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"

    try:
        # Call the Ollama LLM to generate a response
        response = ollama.chat(
            model="llama3.2:3b",  # Replace with your LLM model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": llm_prompt},
            ],
            format="json",
        )
        return response.get("message", {}).get("content", "No response generated")
    except Exception as e:
        st.error(f"Error communicating with the LLM: {e}")
        return "Could not generate a response from the LLM."

# Streamlit UI
st.title("RAG with Vector Search and LLM")

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top K Chunks", min_value=1, max_value=10, value=5, step=1)

query = st.text_input("Enter your query", placeholder="Ask a question...")
if st.button("Search"):
    if query.strip():
        st.write(f"### Query: {query}")

        # Retrieve top K relevant chunks
        with st.spinner("Fetching relevant documents..."):
            retrieved_chunks, metadatas = get_relevant_chunks(query, top_k)

        if retrieved_chunks:
            st.write("### Retrieved Chunks:")
            for i, (chunk, metadata) in enumerate(zip(retrieved_chunks[0], metadatas[0])):
                st.markdown(f"**Chunk {i + 1}:**")
                st.text_area(label="", value=chunk, height=150, key=f"chunk_{i}")
                st.json(metadata, expanded=False)

            # Send user query and retrieved chunks to the LLM
            with st.spinner("Generating response..."):
                llm_response = ask_llm(query, retrieved_chunks[0])

            st.write("### LLM Response:")
            st.markdown(f"> {llm_response}")
        else:
            st.warning("No relevant chunks were retrieved.")
    else:
        st.error("Please enter a query.")
