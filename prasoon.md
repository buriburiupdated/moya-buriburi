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

## Example Usage

```python
from moya.agents.claude_agent import ClaudeAgent, ClaudeAgentConfig

config = ClaudeAgentConfig(
    model_name="claude-3-opus-20240229",
    api_key="YOUR_ANTHROPIC_API_KEY"
)
agent = ClaudeAgent(config)

response = agent.handle_message("What is the weather like today?")
print(response)
```

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


#
#
#
#
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

