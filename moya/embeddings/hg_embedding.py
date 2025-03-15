from moya.embeddings.base_embedding import BaseEmbeddingRepository
from typing import Any, Optional
import os
from langchain_huggingface import HuggingFaceEmbeddings

class HgEmbeddingRepository(BaseEmbeddingRepository):
    def __init__(self, model: str, device: str):
        """
        Initialize the repository with a specific Hugging Face model
        device: cpu or cuda
        """
        #get HF token from environment variable
        token = os.environ("HF_TOKEN")
        if token is None:
            print("Please set the HF_TOKEN environment variable to your Hugging Face API token. Certain models may not be available.")
        super().__init__(embeddings=HuggingFaceEmbeddings(model_name=model, token=token, device=device))
