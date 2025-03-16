# RAG Search Tool
Enables the agent to search through a given knowledge base for documents most similar to a user query.

1. Create an embedding function thst will be used to embed the documents and the query.

    Eg: Using Langchain
    ```python
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    ```

2. Create aa FAISS or ChromaDB vector storw

    Eg: Using FAISS
    ```python
    from moya.vectorstore.faisscpu_vectorstore import FAISSCPUVectorstoreRepository

    path = "path/to/save/vectorstore"
    vector_store = FAISSCPUVectorstoreRepository(path, embeddings)
    vector_store.create_vectorstore()
    ```

    Eg: Using ChromaDB
    ```python
    from moya.vectorstore.chromadb_vectorstore import ChromaDBVectorstoreRepository

    path = "path/to/save/vectorstore"
    vector_store = ChromaDBVectorstoreRepository(path, embeddings)
    vector_store.create_vectorstore()
    ```

3. Now we can use `vector_store.load_file("path/to/file")` to load the documents into the vector store

4. Add the tool to the agent during setup

    ```python
    from moya.vectorsearch.vectorsearchtool import VectorSearchTool

    VectorSearchTool.configure_vector_search_tools(tool_registry)
    ```

5. Now that the vector database is ready, we can search for the `k` most similar queries using `VectorSearchTool.search_vectorstore(query, vectorstrore, k)`

    Eg:
    ```python
    query = "What is the capital of India?"
    k = 5
    results = VectorSearchTool.search_vectorstore(query, vectors_store, 5)
    ```

Check out `examples/quick_start_ollama_rag.py` for a complete example.
