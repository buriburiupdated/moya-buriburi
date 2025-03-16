# ClaudeAgent Documentation

## Introduction
The `ClaudeAgent` is an implementation of an intelligent conversational agent using Anthropic's Claude API. This agent integrates with the **Moya** framework and leverages Claude's ChatCompletion and Completion APIs to generate context-aware responses.

## Key Features
- Utilizes Anthropic's Claude API for conversational capabilities.
- Supports both synchronous and streaming response generation.
- Integrates seamlessly with the **Moya** tool registry for extended functionalities.
- Implements iterative tool-call resolution to enhance user interaction capabilities.

## Class Overview

### `ClaudeAgentConfig`
A configuration class derived from `AgentConfig` that holds the essential parameters for the ClaudeAgent.

**Attributes:**
- `model_name` (str): The Claude model version to use. Default is `"claude-3-opus-20240229"`.
- `api_key` (str): The Anthropic API key, required for authorization.
- `tool_choice` (Optional[str]): Specifies a particular tool choice (if required).

### `ClaudeAgent`
A core agent class that interacts with Claude's API and handles user interactions.

**Methods:**

#### `__init__(self, config: ClaudeAgentConfig)`
Initializes the ClaudeAgent instance.
- **Parameters:**
  - `config`: Configuration instance for the agent.
- **Raises:**
  - `ValueError`: If the API key is not provided.

#### `get_tool_definitions(self) -> List[Dict[str, Any]]`
Retrieves available tool definitions in the format compatible with Claude API.
- **Returns:** A list of tool definitions (each containing `name`, `description`, and `input_schema`).

#### `handle_message(self, message: str, **kwargs) -> str`
Processes a user message and returns the corresponding Claude API response.
- **Parameters:**
  - `message`: The user's input message.
- **Returns:** Response text from the Claude API.

#### `handle_message_stream(self, message: str, stream_callback=None, **kwargs)`
Handles user messages with real-time streaming support.
- **Parameters:**
  - `message`: The user's input message.
  - `stream_callback`: A callback function to handle streaming data.

#### `handle_stream(self, user_message, stream_callback=None, thread_id=None)`
Processes a chat session with streaming functionality.
- **Parameters:**
  - `user_message`: The user's message.
  - `stream_callback`: The callback function for handling streamed text data.
- **Returns:** The response message as a string.

#### `get_response(self, conversation, stream_callback=None)`
Generates a response from the Claude API while supporting streaming responses.
- **Parameters:**
  - `conversation`: List of conversation entries (user/assistant roles).
  - `stream_callback`: Callback function for handling streaming data.
- **Returns:** A dictionary containing `content` (response text) and `tool_calls` (optional).

#### `handle(self, user_message)`
Iteratively resolves tool calls and generates a refined response.
- **Parameters:**
  - `user_message`: The initial input from the user.
- **Returns:** Final response text after tool call processing.

#### `handle_tool_call(self, tool_call)`
Executes tool calls specified in the API response.
- **Parameters:**
  - `tool_call`: Dictionary containing the tool call details such as `id`, `type`, and `function`.
- **Returns:** A string containing the tool's output or a descriptive error message if the tool is unavailable.

## Error Handling
The ClaudeAgent includes robust error handling for API issues:
- API errors are caught and logged with a clear message.
- Invalid tool calls are gracefully handled with a descriptive message.

## Best Practices
- Ensure the API key is securely stored in environment variables to prevent accidental exposure.
- Use the `handle_message_stream()` method for improved performance in real-time chat interfaces.

## Future Enhancements
Potential improvements include:
- Expanded support for additional Claude API features.
- Improved scalability for concurrent requests.

## Conclusion
The `ClaudeAgent` is a powerful integration of Anthropic's Claude API within the **Moya** ecosystem. It supports sophisticated conversational flows and provides robust tool-call resolution for enhanced functionality.

Check out `examples/quick_start_claude.py` for a complete example.

#

# DeepSeekAgent Documentation

## Introduction
The `DeepSeekAgent` is an implementation of an intelligent conversational agent using DeepSeek's API. This agent integrates with the **Moya** framework and leverages DeepSeek's API for text generation.

## Key Features
- Utilizes DeepSeek's API for text generation capabilities.
- Supports both synchronous and streaming response generation.
- Integrates seamlessly with the **Moya** tool registry for extended functionalities.

## Class Overview

### `DeepSeekAgentConfig`
A configuration class derived from `AgentConfig` that holds the essential parameters for the DeepSeekAgent.

**Attributes:**
- `model_name` (str): The DeepSeek model version to use. Default is `"deepseek-coder"`.
- `api_key` (str): The DeepSeek API key, required for authorization.
- `temperature` (float): The temperature setting for the model's response generation. Default is `0.7`.
- `base_url` (str): The base URL for the DeepSeek API. Default is `"https://api.deepseek.com/v1"`.

### `DeepSeekAgent`
A core agent class that interacts with DeepSeek's API and handles user interactions.

**Methods:**

#### `__init__(self, config: DeepSeekAgentConfig)`
Initializes the DeepSeekAgent instance.
- **Parameters:**
  - `config`: Configuration instance for the agent.
- **Raises:**
  - `ValueError`: If the API key is not provided.

#### `_prepare_headers(self) -> Dict[str, str]`
Prepares HTTP headers for API calls.
- **Returns:** A dictionary containing the headers.

#### `_prepare_messages(self, user_input: str) -> List[Dict[str, str]]`
Converts user input to the message format expected by DeepSeek API.
- **Parameters:**
  - `user_input`: The user's input message.
- **Returns:** A list of messages formatted for the DeepSeek API.

#### `handle_message(self, message: str) -> str`
Processes a user message and returns the corresponding DeepSeek API response.
- **Parameters:**
  - `message`: The user's input message.
- **Returns:** Response text from the DeepSeek API.

#### `handle_message_stream(self, message: str) -> Generator[str, None, None]`
Handles user messages with real-time streaming support.
- **Parameters:**
  - `message`: The user's input message.
- **Yields:** Chunks of the model's response.

## Error Handling
The DeepSeekAgent includes robust error handling for API issues:
- API errors are caught and logged with a clear message.
- Invalid responses are gracefully handled with a descriptive message.

## Best Practices
- Ensure the API key is securely stored in environment variables to prevent accidental exposure.
- Use the `handle_message_stream()` method for improved performance in real-time chat interfaces.

## Future Enhancements
Potential improvements include:
- Expanded support for additional DeepSeek API features.
- Improved scalability for concurrent requests.

## Conclusion
The `DeepSeekAgent` is a powerful integration of DeepSeek's API within the **Moya** ecosystem. It supports sophisticated conversational flows and provides robust text generation capabilities for enhanced functionality.

Check out `examples/quick_start_deepseek.py` for a complete example.

#


# HuggingFaceAgent Documentation

## Introduction
The `HuggingFaceAgent` is an intelligent conversational agent leveraging Hugging Face's models and Inference API. This agent integrates with the **Moya** framework and can operate with both local models and the HF API to generate context-aware responses.

## Key Features
- Utilizes Hugging Face's powerful NLP models for various tasks.
- Supports both synchronous and streaming response generation.
- Integrates seamlessly with the **Moya** tool registry for extended functionalities.
- Implements quantization options for efficient local inference.

## Class Overview

### `HuggingFaceAgentConfig`
A configuration class derived from `AgentConfig` that holds the essential parameters for the HuggingFaceAgent.

**Attributes:**
- `model_name` (str): The Hugging Face model version to use. *(Required)*
- `task` (str): Specifies the task type (e.g., `text-generation`, `translation`). *(Required)*
- `access_token` (Optional[str]): The Hugging Face API token for authorization.
- `use_api` (bool): Enables API-based model usage.
- `api_url` (Optional[str]): Custom API URL when `use_api` is `True`.
- `generation_config` (Optional[Dict[str, Any]]): Custom model generation parameters.
- `device` (str): Device for inference (`cpu`, `cuda`).
- `quantization` (Optional[str]): Enables `4bit` or `8bit` quantization for efficiency.

### `HuggingFaceAgent`
A core agent class that interacts with Hugging Face models and handles user interactions.

To create a HuggingFace Agent, you need to provide the following:
```python
    from moya.agents.huggingface_agent import HuggingFaceAgent,     HuggingFaceAgentConfig

    tool_registry = ToolRegistry()

    config = HuggingFaceAgentConfig(
        agent_name="any_name",
        agent_type="HuggingFaceAgent",
        description="Description of the agent",
        model_name="huggingface/repository_name",
        task="task_name (eg: text-generation)",
        tool_registry=tool_registry,
        use_api=False,
        device="cpu",
        generation_config={
            'max_new_tokens': 512,
            'temperature': 0.7,
            'do_sample': True
        }
    )

    agent = HuggingFaceAgent(config)
```

**Methods:**

#### `__init__(self, config: HuggingFaceAgentConfig)`
Initializes the HuggingFaceAgent instance.
- **Parameters:**
  - `config`: Configuration instance for the agent.
- **Raises:**
  - `ValueError`: If `model_name` or `task` are not provided.

#### `handle_message(self, message: str, **kwargs) -> str`
Processes a user message and returns the generated response.
- **Parameters:**
  - `message`: The user's input message.
- **Returns:** Response text from the Hugging Face model.

#### `handle_message_stream(self, message: str, **kwargs) -> Iterator[str]`
Handles user messages with real-time streaming support.
- **Parameters:**
  - `message`: The user's input message.
- **Yields:** Response tokens in real-time.

#### `handle_stream(self, user_message, stream_callback=None, thread_id=None)`
Processes a chat session with streaming functionality.
- **Parameters:**
  - `user_message`: The user's message.
  - `stream_callback`: The callback function for handling streamed text data.
- **Returns:** The response message as a string.

#### `get_response(self, conversation, stream_callback=None)`
Generates a response using the Hugging Face model.
- **Parameters:**
  - `conversation`: List of conversation entries (user/assistant roles).
  - `stream_callback`: Callback function for handling streaming data.
- **Returns:** A dictionary containing `content` (response text) and `tool_calls` (optional).

#### `handle_tool_call(self, tool_call)`
Executes tool calls specified in the API response.
- **Parameters:**
  - `tool_call`: Dictionary containing tool call details such as `id`, `type`, and `function`.
- **Returns:** A string containing the tool's output or a descriptive error message if the tool is unavailable.

## Error Handling
The HuggingFaceAgent includes robust error handling:
- Errors in model calls are caught and logged with descriptive messages.
- Invalid tool calls are managed gracefully with appropriate error handling.

## Best Practices
- Use `quantization` for faster and more efficient inference on local devices.
- Utilize the `handle_message_stream()` method for improved performance in real-time chat interfaces.

## Future Enhancements
Potential improvements include:
- Expanded support for additional model architectures.
- Improved scalability for concurrent requests.

## Conclusion
The `HuggingFaceAgent` is a powerful integration of Hugging Face's versatile NLP models within the **Moya** ecosystem. It supports flexible conversational flows, real-time streaming, and efficient local inference.

Check out `examples/quick_start_hf_agent.py` for a complete example.



#

# Text Autocomplete Tool for Moya

## Overview
The **Text Autocomplete** tool is a lightweight text completion assistant designed to predict the next **few words** of a given sentence. Integrated with the **Moya** framework, this tool leverages the **OllamaAgent** powered by the locally hosted language models for effective autocompletion capabilities.

## Key Features
- Predicts **5-6 words** maximum to extend a given sentence.
- Ensures completion aligns with the user's writing style and tone.
- Focuses strictly on sentence continuation without initiating new ideas.
- Provides grammatically correct and contextually meaningful outputs.

## Configuration Details
The tool uses the following configuration:
- **Model** (in-use): `mistral:latest` for effective text completion.
- **Temperature**: Set to `0.3` for controlled and stable predictions.
- **Base URL**: Configured as `http://localhost:11434` to align with local Ollama API endpoints.
- **Context Window**: Limited to `1024` tokens for optimal performance.

## Usage Workflow
### Step 1: Initializing the Agent
- The tool initializes a shared instance of the **OllamaAgent** using the specified configuration.
- During initialization, the tool validates the agent by sending a test query.

### Step 2: Cleaning Input Text
- The input text is cleaned by:
  - Removing conversational markers like "User:" or "Bot:".
  - Stripping trailing punctuation marks like periods and spaces to enhance response accuracy.

### Step 3: Text Completion
- If the cleaned input matches the previously processed text, the tool returns the cached result.
- Otherwise, the tool queries the **OllamaAgent** and limits the output to 5-6 words.
- The result is further refined by removing incomplete words or punctuation fragments for clarity.

### Step 4: Caching for Efficiency
- The tool stores the most recent input and output to minimize redundant API calls and improve performance.

## Error Handling
- If the **OllamaAgent** fails to respond, an appropriate error message is returned.
- Errors during text cleaning or completion are gracefully handled to ensure stability.

## Best Practices
- Ensure the **OllamaAgent** server is running on `http://localhost:11434` before initializing the tool.
- For improved performance, use concise yet informative input text to guide accurate completions.

## Future Enhancements
- Expanded language model options for enhanced performance.
- Additional cleaning rules to improve response clarity.

## Conclusion
The **Text Autocomplete** tool simplifies sentence continuation tasks by predicting concise and meaningful completions. With its integration into the **Moya** framework, it offers a powerful solution for enhancing text prediction workflows.

Check out `examples/text-completion` for a complete example.

#

# RAG Search Tool
Enables the agent to search through a given knowledge base for documents most similar to a user query.

1. Create an embedding function thst will be used to embed the documents and the query.

    Eg: Using Langchain
    ```python
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    ```

2. Create aa FAISS or ChromaDB vector storw

    Eg: Using FAISS
    ```python
    from moya.vectorstore.faisscpu_vectorstore import FAISSCPUVectorstoreRepository

    path = "path/to/save/vectorstore"
    vector_store = FAISSCPUVectorstoreRepository(path, embeddings)
    vector_store.create_vectorstore()
    ```

    Eg: Using ChromaDB
    ```python
    from moya.vectorstore.chromadb_vectorstore import ChromaDBVectorstoreRepository

    path = "path/to/save/vectorstore"
    vector_store = ChromaDBVectorstoreRepository(path, embeddings)
    vector_store.create_vectorstore()
    ```

3. Now we can use `vector_store.load_file("path/to/file")` to load the documents into the vector store

4. Add the tool to the agent during setup

    ```python
    from moya.vectorsearch.vectorsearchtool import VectorSearchTool

    VectorSearchTool.configure_vector_search_tools(tool_registry)
    ```

5. Now that the vector database is ready, we can search for the `k` most similar queries using `VectorSearchTool.search_vectorstore(query, vectorstrore, k)`

    Eg:
    ```python
    query = "What is the capital of India?"
    k = 5
    results = VectorSearchTool.search_vectorstore(query, vectors_store, 5)
    ```

Check out `examples/quick_start_ollama_rag.py` for a complete example.
#

# Dynamic RAG

By leveratng the `VectorSearchTools` and `SearchTools` we can create a agent that searches knowledge base for the answer to the question and also use web search to verify answers to provide more accurate answers. 

Check out `examples/quick_start_dynamic_rag.py` for how to implement this.

#

# BidOrchestrator Documentation

## Introduction
The `BidOrchestrator` represents a dynamic approach to agent orchestration, using a marketplace model where agents compete and collaborate to handle user requests.

## Core Concepts
- **Bidding System**: Agents bid on user requests with confidence scores.
- **Dynamic Team Formation**: Creates teams when multiple agents would be beneficial.
- **Performance Learning**: Improves over time by tracking agent performance.

## Workflow

### 1. Bid Collection
When a user message arrives, the orchestrator:
- Requests confidence scores from all available agents.
- Can run in parallel for faster response.
- For agents without a `bid_on_task` method, it estimates confidence based on:
  - Keyword overlap between message and agent description.
  - Historical performance of the agent.

### 2. Agent Selection
The orchestrator makes a key decision:
- Use a single agent when:
  - One agent has significantly higher confidence (>0.7 and 0.2+ higher than others).
  - Only one agent is available.
- Form a team when:
  - Multiple agents have similar high confidence scores.
  - No single agent is highly confident (below the team threshold).

### 3. Processing
- For single agent:
  - Simply passes the request to the chosen agent.
  - Supports streaming responses.
  - Tracks performance metrics.
- For team-based processing:
  - Collects responses from all team members in parallel.
  - Synthesizes responses into a coherent answer.
  - If available, uses a dedicated synthesis agent.
  - Otherwise, combines the highest confidence response with insights from others.

### 4. Performance Learning
The orchestrator continuously updates performance metrics for agents:
- Tracks average response time.
- Maintains success rates (could be updated with user feedback).
- Uses this data to influence future agent selection.

## Configuration Parameters
Key parameters that control behavior:
- `min_confidence`: Minimum confidence needed for an agent to handle a request.
- `team_threshold`: Confidence threshold for team formation.
- `max_team_size`: Maximum number of agents in a team.
- `parallel_bidding`: Whether to collect bids in parallel.

## Bidding Algorithm
The confidence calculation combines:
- Base confidence score (0.1).
- Word overlap between user query and agent description (up to 0.5).
- Historical performance factor (up to 0.4).

This creates a balanced approach where teams form only when genuinely beneficial, rather than for every request.

Check out `examples/quick_start_bid_orchestrator.py` for a complete example.

#

# Dynamic Tool Registration
## Overview
The Dynamic Tool Registration feature in Moya allows you to create and register custom tools at runtime, extending agent capabilities without modifying the framework code. This powerful functionality allows users to add their own tool makes it possible to add domain-specific functions as tools that agents can discover and call during conversations.

## Key Capabilities
- **Runtime Tool Registration:** Register new tools during program execution
- **Function-Based Tools:** Convert Python functions into agent-callable tools
- **Auto-Documentation:** Automatically generates tool descriptions from function docstrings
- **Parameter Validation:** Ensures tool parameters are properly formatted for LLM consumption

## How to Use Dynamic Tool Registration
### Basic Setup

1) Import the necessary components:
```python
from moya.tools.tool_registry import ToolRegistry
from moya.tools.dynamic_tool_registrar import DynamicToolRegistrar
```

2) Create a tool registry and register your custom functions:
```python
# Create a tool registry
tool_registry = ToolRegistry()

# Register a custom function
DynamicToolRegistrar.register_function_as_tool(
    tool_registry=tool_registry,
    function=your_custom_function
)
```

3) Configure your agent with the tool registry:
```python
agent = YourAgent(
    agent_config=AgentConfig(
        system_prompt="You have access to custom tools...",
        tool_registry=tool_registry,
        # other configuration...
    )
)
```

### Example Usage
Here's a complete example of using dynamic tool registration with an Ollama agent:
```python
from moya.tools.tool_registry import ToolRegistry
from moya.tools.dynamic_tool_registrar import DynamicToolRegistrar
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.agents.ollama_agent import OllamaAgent, AgentConfig

# Define custom functions to be registered as tools
def temperature_converter(celsius: float = None, fahrenheit: float = None) -> dict:
    """Convert between Celsius and Fahrenheit.
    
    Parameters:
        - celsius: Temperature in Celsius to convert to Fahrenheit
        - fahrenheit: Temperature in Fahrenheit to convert to Celsius
    
    Returns a dictionary with both temperature values.
    """
    result = {}
    
    if celsius is not None:
        result["celsius"] = float(celsius)
        result["fahrenheit"] = (float(celsius) * 9/5) + 32
    elif fahrenheit is not None:
        result["fahrenheit"] = float(fahrenheit)
        result["celsius"] = (float(fahrenheit) - 32) * 5/9
    else:
        return {"error": "Please provide either celsius or fahrenheit value"}
        
    return result

# Set up agent with dynamic tools
def setup_agent():
    # Create a tool registry
    tool_registry = ToolRegistry()
    
    # Register user functions as tools
    DynamicToolRegistrar.register_function_as_tool(
        tool_registry=tool_registry,
        function=temperature_converter
    )
    
    # Create agent config with tools
    config = AgentConfig(
        agent_name="custom_tools_agent",
        agent_type="OllamaAgent",
        description="An agent with custom temperature conversion tools",
        system_prompt="You are a helpful assistant with access to temperature conversion tools.",
        tool_registry=tool_registry,
        llm_config={
            'model_name': "llama3.1:latest",
            'base_url': "http://localhost:11434"
        }
    )
    
    # Create the agent
    agent = OllamaAgent(agent_config=config)
    
    # Set up orchestrator
    agent_registry = AgentRegistry()
    agent_registry.register_agent(agent)
    orchestrator = SimpleOrchestrator(
        agent_registry=agent_registry,
        default_agent_name="custom_tools_agent"
    )
    
    return orchestrator, agent
```

### Advanced Features
#### Creating Tools from Code Strings

You can also create tools from code strings, which is useful for scenarios where you want to allow users or other systems to define tools:

```python
code_string = """
def calculate_area(length: float, width: float) -> float:
    \"\"\"Calculate the area of a rectangle.
    
    Parameters:
        - length: The length of the rectangle
        - width: The width of the rectangle
    \"\"\"
    return length * width
"""

DynamicToolRegistrar.register_from_code(
    tool_registry=tool_registry,
    function_code=code_string,
    tool_name="AreaCalculator",
    description="Calculates the area of a rectangle"
)
```

#### Registering Multiple Functions at Once
```python
DynamicToolRegistrar.register_functions(
    tool_registry=tool_registry,
    functions=[function1, function2, function3],
    names=["Tool1", "Tool2", "Tool3"],  # Optional
    descriptions=["Description1", "Description2", "Description3"]  # Optional
)
```

Check out `examples/quick_start_ollama_dynamic_tool.py` for a complete example of using dynamic tool registration with an Ollama agent.

#

# Search Tool

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

#### `test_azure_openai.py`

This example demonstrates how to set up and use the `AzureOpenAI` agent with web search capabilities.

#### `test_azure_openai_dynamic_rag.py`

This example demonstrates how to set up and use the `AzureOpenAI` agent with dynamic RAG (Retrieval-Augmented Generation) and web search capabilities.

#### `quick_start_claude_search.py`

This example uses claude api to integrate web search capabilities.

