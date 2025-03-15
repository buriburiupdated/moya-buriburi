"""
Chroma vectorstore repository for Moya

An implementation of BaseVectorstoreRepository that uses the Chroma library to store and retrieve vectors
"""
from moya.vectorstore.base_vectorstore import BaseVectorstoreRepository
from langchain_chroma import Chroma
from langchain_core.documents.base import Document

class ChromaVectorstoreRepository(VectorstoreRepository):
    def __init__(self, path, embeddings):
        """
        Initialize the repository with a specific path and embedding model
        """
        self.embedding = embeddings
        self.path = path
        self.vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db-nomic")

    def add_vector(self, chunks: List[Document]) -> None:
        """
        Add a new vector to the vectorstore
        """
        self.vectorstore.add_documents(chunks)
        self.vectorstore.save_local(self.path)
    
    def get_context(self, query: str, k: int) -> List[Document]:
        """
        Retrieve the k closest vectors to the query
        """
        results = self.vectorstore.similarity_search(query,k)
        if not results:
            return []
        return results