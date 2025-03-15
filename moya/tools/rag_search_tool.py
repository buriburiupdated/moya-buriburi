"""
Vector search tool implementation for Moya.
Provides RAG capabilities by searching through vector databases.
"""

import json
from typing import Dict, List, Any, Optional
from moya.tools.base_tool import BaseTool

class VectorSearchTool:
    """Tools for vector database search capabilities."""
    def __init__(self):
        self.vector_store = None
    
    @staticmethod
    def search_vectorstore(self, query: str, collection_name: str = "faiss-index", k: int = 5) -> str:
        """
        Search a vector database for relevant documents based on semantic similarity.
        
        Args:
            query: The search query text
            collection_name: Name of the vector collection to search in
            k: Number of results to return
            
        Returns:
            JSON string containing search results
        """
        try:
            # Get the appropriate vector store
            # In a real implementation, you might want to maintain a registry of vector stores
            vector_store = self.vector_store
            
            # Search for relevant documents
            results = vector_store.get_context(query, k)
            
            # Format results
            formatted_results = []
            for i, doc in enumerate(results):
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, "score", None)
                })
            
            return json.dumps({
                "query": query,
                "collection": collection_name,
                "results": formatted_results
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "query": query,
                "collection": collection_name
            })
    
    @staticmethod
    def configure_vector_search_tools(tool_registry, vector_store) -> None:
        """
        Configure vector search tools and register them with the tool registry.
        
        Args:
            tool_registry: The tool registry to register tools with.
        """
        VectorSearchTool.vector_store = vector_store
        tool_registry.register_tool(
            BaseTool(
                name="VectorSearchTool",
                function=VectorSearchTool.search_vectorstore,
                description="Search a vector database for semantically similar content",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "The search query",
                        "required": True
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the vector collection to search (default: 'faiss-index')",
                        "required": False
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "required": False
                    }
                }
            )
        )