from moya.embeddings.base_embedding import BaseEmbeddingRepository
from typing import Any, Optional
from sentence_transformers import SentenceTransformer

class HgEmbeddingRepository(BaseEmbeddingRepository):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode_text(self, text: str) -> Any:
        """Encode text into an embedding"""
        return self.model.encode(text)

    def encode_texts(self, texts: list[str]) -> list[Any]:
        """Encode a list of texts into embeddings"""
        return self.model.encode(texts)

    def get_model_config(self) -> dict:
        """Get the model's configuration"""
        return self.model[0].auto_model.config.to_dict()

