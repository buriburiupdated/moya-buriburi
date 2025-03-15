from moya.embeddings.base_embedding import BaseEmbeddingRepository
from typing import Any, Optional
from sentence_transformers import SentenceTransformer

class HgEmbeddingRepository(BaseEmbeddingRepository):
    def __init__(self, model: Optional[str] = None):
        available_models = [
            'all-MiniLM-L6-v2',
            'paraphrase-MiniLM-L6-v2',
            'distiluse-base-multilingual-cased-v1'
        ]
        if model is None:
            print("Available models:")
            for idx, m in enumerate(available_models, 1):
                print(f"{idx}. {m}")
            choice = input("Choose a model by number or press enter to use the default (1): ")
            if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                model = available_models[int(choice) - 1]
            else:
                model = available_models[0]
        self.model = SentenceTransformer(model)

    def encode_text(self, text: str) -> Any:
        """Encode text into an embedding"""
        return self.model.encode(text)

    def encode_texts(self, texts: list[str]) -> list[Any]:
        """Encode a list of texts into embeddings"""
        return self.model.encode(texts)

    def get_model_config(self) -> dict:
        """Get the model's configuration"""
        return self.model[0].auto_model.config.to_dict()

