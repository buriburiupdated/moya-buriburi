# Web Search Tool
Enables the agent to search the web for the most relevant information based on a user query.

1. Add the tool to the agent during setup

    ```python
    from moya.search.searchtool import SearchTool


    def setup_agent():
      tool_registry = ToolRegistry()
      SearchTool.configure_search_tools(tool_registry)
      # Rest of the setup...
    ```

2. Now we can search the web for the most relevant information using `SearchTool.search_web_free(query)` (for searches without the need of an API) or \
 `SearchTool.search_web(query)` (for better results provided the `SERPAPI_KEY` is set in the environment variables).



Check out `examples/quick_start_web_search.py` for a complete example.

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