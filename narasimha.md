# Usage Instructions

## Using `quick_start_deepseek`

1. Add the API key as:
    ```bash
    export DEEPSEEK_API_KEY=your-new-api-key-here
    ```
2. Run from the root directory:
    ```bash
    python examples/quick_start_deepseek.py
    ```
3. Use the terminal to interact just like the other examples.

## Using `quick_start_ollama_deepseek`

1. In the root directory, run:
    ```bash
    python examples/quick_start_ollama_deepseek.py
    ```
2. Interact with the terminal.

## Using `test_azure_openai`

1. First, configure the OpenAI Azure keys as follows:
    ```bash
    export AZURE_OPENAI_API_KEY="your-api-key-here"
    export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
    export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
    ```
2. Run:
    ```bash
    python examples/test_azure_openai.py
    ```
3. In the terminal, type something like:
    ```bash
    who is the number 9 for real madrid? rely on websearch
    ```
4. The top 5 results along with the conclusion the model makes will be displayed. For example:
    ```
    You: who is the number 9 for real madrid? rely on websearch
    Assistant: <search query="current number 9 for Real Madrid 2025">
    Performing searches based on assistant's request...
    [SEARCH] Searching for: current number 9 for Real Madrid 2025
    ...
    Assistant: The number 9 for Real Madrid in 2025 is Kylian Mbappe. 
    Source: Real Madrid Latest News
    ```

## Using `test_azure_openai_dynamic_rag.py`

1. Make sure you have added the API keys like in the previous use case.
2. Remove any existing vector store:
    ```bash
    rm -rf azure-dynamic-faiss
    ```
3. In the root directory, run:
    ```bash
    python examples/test_azure_openai_dynamic_rag.py
    ```
4. Search something in the terminal, for example:
    ```bash
    /search president of united states 2025
    ```
5. In the terminal, you will see web results, and if you go to `azure-dynamic-faiss/index.pkl`, you can see that the web results are getting added to the knowledge base.
6. Type something like:
    ```bash
    /kb president of the united states
    ```
    The terminal will search the knowledge base and return the information.
7. The model can search the knowledge base as well as the internet, combining RAG and web crawl.

# Detailed Documentation

## Agents and Tools

### DeepSeek Agent

The `DeepSeekAgent` is an implementation of an intelligent conversational agent using DeepSeek's API. This agent integrates with the **Moya** framework and leverages DeepSeek's API for text generation.

**Key Features:**
- Utilizes DeepSeek's API for text generation capabilities.
- Supports both synchronous and streaming response generation.
- Integrates seamlessly with the **Moya** tool registry for extended functionalities.

**Class Overview:**

#### `DeepSeekAgentConfig`
A configuration class derived from `AgentConfig` that holds the essential parameters for the DeepSeekAgent.

**Attributes:**
- `model_name` (str): The DeepSeek model version to use. Default is `"deepseek-coder"`.
- `api_key` (str): The DeepSeek API key, required for authorization.
- `temperature` (float): The temperature setting for the model's response generation. Default is `0.7`.
- `base_url` (str): The base URL for the DeepSeek API. Default is `"https://api.deepseek.com/v1"`.

#### `DeepSeekAgent`
A core agent class that interacts with DeepSeek's API and handles user interactions.

**Methods:**

- `__init__(self, config: DeepSeekAgentConfig)`: Initializes the DeepSeekAgent instance.
- `_prepare_headers(self) -> Dict[str, str]`: Prepares HTTP headers for API calls.
- `_prepare_messages(self, user_input: str) -> List[Dict[str, str]]`: Converts user input to the message format expected by DeepSeek API.
- `handle_message(self, message: str) -> str`: Processes a user message and returns the corresponding DeepSeek API response.
- `handle_message_stream(self, message: str) -> Generator[str, None, None]`: Handles user messages with real-time streaming support.

### Search Tool

The `SearchTool` provides web search capabilities for agents. It supports both paid (SerpAPI) and free (DuckDuckGo) search options.

**Key Features:**
- Performs web searches using SerpAPI or DuckDuckGo.
- Returns formatted search results in JSON format.
- Integrates with the **Moya** tool registry.

**Methods:**

- `search_web(query: str, num_results: int = 5) -> str`: Searches the web using SerpAPI.
- `search_web_free(query: str, num_results: int = 5) -> str`: Searches the web using DuckDuckGo (no API key required).
- `configure_search_tools(tool_registry) -> None`: Configures search tools and registers them with the tool registry.

### Example Usage

#### `quick_start_deepseek.py`

This example demonstrates how to set up and use the `DeepSeekAgent` with conversation memory.

#### `quick_start_ollama_deepseek.py`

This example demonstrates how to set up and use the `OllamaAgent` with the `DeepSeek-Coder` model and conversation memory.

#### `test_azure_openai.py`

This example demonstrates how to set up and use the `AzureOpenAI` agent with web search capabilities.

#### `test_azure_openai_dynamic_rag.py`

This example demonstrates how to set up and use the `AzureOpenAI` agent with dynamic RAG (Retrieval-Augmented Generation) and web search capabilities.

## File Descriptions

### /moya/tools/search_tool.py

This file implements the `SearchTool` class, which provides web search capabilities for agents. It supports both paid (SerpAPI) and free (DuckDuckGo) search options. The class includes methods to perform web searches and return formatted search results in JSON format. It also includes a method to configure and register these search tools with a tool registry.

### moya/agents/deepseek_agent.py

This file implements the `DeepSeekAgent` class, which is an agent that uses DeepSeek's API for text generation. The agent can handle both synchronous and streaming responses. It includes methods to prepare HTTP headers and messages for API calls, process user messages, and handle message streams. The agent configuration is stored in the `AgentConfig` class.

### /hexamples/test_azure_openai.py

This file provides an interactive Azure OpenAI chat interface with web search capability. It includes functions to perform web searches, create an Azure OpenAI client, and run an interactive chat session. The chat session can handle user inputs, perform searches based on specific commands, and integrate search results into the conversation.

### /examples/test_azure_openai_dynamic_rag.py

This file provides an interactive Azure OpenAI chat interface with dynamic Retrieval-Augmented Generation (RAG) and web search capabilities. It combines features from `test_azure_openai.py` and `quick_start_dynamic_rag.py`. It includes functions to set up a vector store, add text to the knowledge base, create an Azure OpenAI client, extract and add search information to the knowledge base, and run an interactive chat session with dynamic RAG.

### /examples/quick_start_ollama_deepseek.py

This file provides an interactive chat example using the Ollama agent with the DeepSeek-Coder model and conversation memory. It includes functions to set up the agent, format conversation context, and run an interactive chat session. The chat session can handle user inputs, store messages in memory, and provide responses using the DeepSeek-Coder model.

### /examples/quick_start_deepseek.py

This file provides an interactive chat example using the DeepSeek agent with conversation memory. It includes functions to set up the agent, format conversation context, and run an interactive chat session. The chat session can handle user inputs, store messages in memory, and provide responses using the DeepSeek model.

