from moya.embeddings.hg_embedding import HgEmbeddingRepository
from moya.embeddings.ollama_embedding import OllamaEmbeddingRepository
from moya.embeddings.openai_embedding import OpenAIEmbeddingRepository
import os
import dotenv
dotenv.load_dotenv()

def test_hg_embedding():
    repo = HgEmbeddingRepository()
    
    # Test single text encoding
    text = "Hello, world!"
    embedding = repo.encode_text(text)
    print("Single text embedding:", embedding)
    
    # Test multiple texts encoding
    texts = ["Hello, world!", "How are you?"]
    embeddings = repo.encode_texts(texts)
    print("Multiple texts embeddings:", embeddings)
    
    # Test model configuration retrieval
    config = repo.get_model_config()
    print("Model configuration:", config)

def test_ollama_embedding():
    repo = OllamaEmbeddingRepository()
    
    # Test single text encoding
    text = "Hello, world!"
    embedding = repo.encode_text(text)
    print("Single text embedding:", embedding)
    
    # Test multiple texts encoding
    texts = ["Hello, world!", "How are you?"]
    embeddings = repo.encode_texts(texts)
    print("Multiple texts embeddings:", embeddings)
    
    # Test model configuration retrieval
    config = repo.get_model_config()
    print("Model configuration:", config)

def test_openai_embedding(api_key, endpoint, deployment_name="text-embedding-ada-002"):
    if not api_key or not endpoint:
        print("Error: No API key or endpoint provided. Please set the OPEN_AI and AZURE_ENDPOINT environment variables.")
        return
    
    repo = OpenAIEmbeddingRepository(api_key, endpoint, deployment_name)
    
    # Test single text encoding
    text = "Hello, world!"
    embedding = repo.encode_text(text)
    print("Single text embedding:", embedding)
    
    # Test multiple texts encoding
    texts = ["Hello, world!", "How are you?"]
    embeddings = repo.encode_texts(texts)
    print("Multiple texts embeddings:", embeddings)
    
    # Test model configuration retrieval
    config = repo.get_model_config()
    print("Model configuration:", config)

if __name__ == "__main__":
    api_key = os.getenv("OPEN_AI")
    endpoint = os.getenv("AZURE_ENDPOINT")
    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "text-embedding-ada-002")
    test_openai_embedding(api_key, endpoint, deployment_name)
