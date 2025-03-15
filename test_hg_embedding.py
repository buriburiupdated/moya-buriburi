from moya.embeddings.hg_embedding import HgEmbeddingRepository

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

if __name__ == "__main__":
    test_hg_embedding()
