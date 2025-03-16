# Problem 1
Issue with ollama local. Moved to branch kannan/fix-tooling
Tried
```
pip install .
```

# Problem 2
The `crewai-tools` package breaks a lot of dependencies. \
Removed `crewai-tools` from `pyproject.toml`

# Problem 3
Issue with dotenv. Replaced dotenv with python-dotenv in .toml file

# Problem 4
Issue with adding support for openai embeddings due to unavailability of proper API keys.

# Work done

### Added support for claude and deepseek.

### Added support for hugging face agent

### Implemented different types of embeddings

### Implemented different types of vector storage

### Implemented RAG tools and RAG

### Implemented text completions using ollama's locally hosted models

### Added support for web search
