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