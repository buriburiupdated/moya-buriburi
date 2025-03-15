"""
Base repository for embeddings in Moya

Defines an abstract class describing methods for managing vectors across various databases.
"""

import abc
from typing import Any, Optional

class BaseEmbeddingRepository(abc.ABC):
    """
    Abstract interface for storing and retrieving embeddings
    """
    def __init__(self, embeddings: any):
        """
        Initialize the repository with a specific embedding model
        """
        self.embeddings = embeddings

    @abc.abstractmethod
    def encode_text(self, text: str) -> Any:
        """
        Encode text into an embedding
        """
        pass

    @abc.abstractmethod
    def encode_texts(self, texts: list[str]) -> list[Any]:
        """
        Encode a list of texts into embeddings
        """
        pass

    @abc.abstractmethod
    def get_model_config(self) -> dict:
        """
        Get the model's configuration
        """
        pass