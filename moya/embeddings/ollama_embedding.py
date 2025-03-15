from moya.embeddings.base_embedding import BaseEmbeddingRepository
from typing import Any, Optional
import ollama
from langchain_ollama import OllamaEmbeddings

class OllamaEmbeddingRepository(BaseEmbeddingRepository):
    def __init__(self, model: Optional[str] = None):
        """
        Initialize the repository with a specific Ollama model
        """
        # Get available models directly from Ollama's model list
        available_models = [m['model'] for m in ollama.list()['models']]
        print("\nTo get more models, use: ollama pull <model-name>\n")
        if model is None:
            print("Available models:")
            for idx, m in enumerate(available_models, 1):
                print(f"{idx}. {m}")
            choice = input("\nChoose a model by number, type model name, or press Enter for default (1): ").strip()

            if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                model = available_models[int(choice) - 1]
            elif choice:  # If user types a model name directly
                model = choice
            else:
                model = available_models[0]
        self.model = model
        super().__init__(embeddings=OllamaEmbeddings(model=model))
        print(f"Selected model: {self.model}")
