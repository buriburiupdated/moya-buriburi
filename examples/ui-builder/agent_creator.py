"""
Moya Agent Builder - Visual UI for creating agent systems

This tool provides a graphical interface for configuring agents, tools, and orchestrators
and generating the corresponding code.
"""

import streamlit as st
import json
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Moya Agent Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
    }
    .agent-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .tool-card {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .orchestrator-card {
        background-color: #f8f0ff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .title-section {
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agents' not in st.session_state:
    st.session_state.agents = []
if 'selected_tools' not in st.session_state:
    st.session_state.selected_tools = []
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'code_generated' not in st.session_state:
    st.session_state.code_generated = False

# Title section
st.markdown("<div class='title-section'>", unsafe_allow_html=True)
st.title("Moya Agent Builder")
st.markdown("Create your agent system visually and generate code automatically")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar for adding agents and general settings
with st.sidebar:
    st.header("Moya Configuration")
    
    # Add Agent section
    st.subheader("Add New Agent")
    agent_type = st.selectbox(
        "Agent Type", 
        ["OllamaAgent", "OpenAIAgent", "AzureOpenAIAgent", "BedrockAgent"]
    )
    
    agent_name = st.text_input("Agent Name", "")
    agent_description = st.text_area("Description", "", height=100)
    system_prompt = st.text_area("System Prompt", "", height=150)
    
    # Model selection based on agent type
    if agent_type == "OllamaAgent":
        model_options = ["llama3", "llama3.1:latest", "codellama", "mistral"]
        base_url = st.text_input("Base URL", "http://localhost:11434")
    elif agent_type == "OpenAIAgent":
        model_options = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        api_key = st.text_input("API Key", "", type="password")
    elif agent_type == "AzureOpenAIAgent":
        model_options = ["gpt-4", "gpt-3.5-turbo"]
        api_key = st.text_input("API Key", "", type="password")
        api_base = st.text_input("API Base", "")
        api_version = st.text_input("API Version", "2023-05-15")
    elif agent_type == "BedrockAgent":
        model_options = ["anthropic.claude-v2", "amazon.titan-text"]
        region = st.text_input("Region", "us-east-1")
    
    model_name = st.selectbox("Model Name", model_options)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    # Add agent button
    if st.button("Add Agent"):
        if agent_name and agent_description:
            agent_config = {
                "agent_type": agent_type,
                "name": agent_name,
                "description": agent_description,
                "system_prompt": system_prompt,
                "model_name": model_name,
                "temperature": temperature
            }
            
            # Add agent-specific configs
            if agent_type == "OllamaAgent":
                agent_config["base_url"] = base_url
            elif agent_type == "OpenAIAgent":
                agent_config["api_key"] = api_key
            elif agent_type == "AzureOpenAIAgent":
                agent_config["api_key"] = api_key
                agent_config["api_base"] = api_base
                agent_config["api_version"] = api_version
            elif agent_type == "BedrockAgent":
                agent_config["region"] = region
            
            st.session_state.agents.append(agent_config)
            st.success(f"Added agent: {agent_name}")
        else:
            st.error("Agent name and description are required")

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Agents", "Tools", "Orchestrator", "Generated Code"])

# Tab 1: Agents
with tab1:
    st.header("Configured Agents")
    if not st.session_state.agents:
        st.info("No agents added yet. Use the sidebar to add agents.")
    else:
        for i, agent in enumerate(st.session_state.agents):
            with st.container():
                st.markdown(f"<div class='agent-card'>", unsafe_allow_html=True)
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"{agent['name']} ({agent['agent_type']})")
                    st.write(f"**Description:** {agent['description']}")
                    st.write(f"**Model:** {agent['model_name']}")
                    with st.expander("System Prompt"):
                        st.text(agent['system_prompt'])
                with col2:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.agents.pop(i)
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Tools
with tab2:
    st.header("Select Tools")
    
    # Memory tools
    memory_tools = st.checkbox("Memory Tools (EphemeralMemory)", value=True)
    if memory_tools:
        st.session_state.selected_tools = ["memory_tools"] if "memory_tools" not in st.session_state.selected_tools else st.session_state.selected_tools
    else:
        if "memory_tools" in st.session_state.selected_tools:
            st.session_state.selected_tools.remove("memory_tools")
    
    # Agent Consultation Tool
    consultation_tool = st.checkbox("Agent Consultation Tool", value=True)
    if consultation_tool:
        st.session_state.selected_tools = ["consultation_tool"] if "consultation_tool" not in st.session_state.selected_tools else st.session_state.selected_tools
    else:
        if "consultation_tool" in st.session_state.selected_tools:
            st.session_state.selected_tools.remove("consultation_tool")
            
    # RAG Search Tool
    rag_tool = st.checkbox("RAG Search Tool (Vector Search)")
    if rag_tool:
        with st.container():
            st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
            st.subheader("RAG Search Configuration")
            vectorstore_type = st.selectbox("Vector Store Type", ["FAISS", "Chroma", "Pinecone"])
            embeddings_model = st.selectbox("Embeddings", ["OpenAI", "HuggingFace", "Ollama"])
            data_source = st.text_input("Data Source Path (documents)", "")
            st.markdown("</div>", unsafe_allow_html=True)
        if "rag_tool" not in st.session_state.selected_tools:
            st.session_state.selected_tools.append("rag_tool")
    else:
        if "rag_tool" in st.session_state.selected_tools:
            st.session_state.selected_tools.remove("rag_tool")
    
    # Web Search Tool
    web_search = st.checkbox("Web Search Tool")
    if web_search:
        with st.container():
            st.markdown("<div class='tool-card'>", unsafe_allow_html=True)
            st.subheader("Web Search Configuration")
            search_engine = st.selectbox("Search Engine", ["Google", "Bing", "DuckDuckGo"])
            api_key_search = st.text_input("API Key (if applicable)", "", type="password")
            st.markdown("</div>", unsafe_allow_html=True)
        if "web_search" not in st.session_state.selected_tools:
            st.session_state.selected_tools.append("web_search")
    else:
        if "web_search" in st.session_state.selected_tools:
            st.session_state.selected_tools.remove("web_search")
            
    # Math Tool
    math_tool = st.checkbox("Math Tool")
    if math_tool:
        if "math_tool" not in st.session_state.selected_tools:
            st.session_state.selected_tools.append("math_tool")
    else:
        if "math_tool" in st.session_state.selected_tools:
            st.session_state.selected_tools.remove("math_tool")

# Tab 3: Orchestrator
with tab3:
    st.header("Select Orchestrator")
    
    orchestrator_type = st.radio(
        "Orchestrator Type",
        ["SimpleOrchestrator", "MultiAgentOrchestrator", "BidOrchestrator", "ReactOrchestrator"]
    )
    
    st.markdown("<div class='orchestrator-card'>", unsafe_allow_html=True)
    if orchestrator_type == "SimpleOrchestrator":
        st.write("**Simple Orchestrator**: Routes all messages to a default agent.")
        default_agent = st.selectbox("Default Agent", [agent["name"] for agent in st.session_state.agents] if st.session_state.agents else ["No agents available"])
    
    elif orchestrator_type == "MultiAgentOrchestrator":
        st.write("**Multi-Agent Orchestrator**: Routes messages to different agents based on a classifier.")
        classifier_type = st.selectbox("Classifier", ["LLMClassifier", "KeywordClassifier", "RuleBasedClassifier"])
        default_agent = st.selectbox("Default Agent", [agent["name"] for agent in st.session_state.agents] if st.session_state.agents else ["No agents available"])
    
    elif orchestrator_type == "BidOrchestrator":
        st.write("**Bid Orchestrator**: Agents bid on tasks and the orchestrator selects the most confident agent.")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
        team_mode = st.checkbox("Enable Team Mode", value=True)
        
    elif orchestrator_type == "ReactOrchestrator":
        st.write("**ReAct Orchestrator**: Implements the Reasoning and Acting paradigm for more deliberate decision making.")
        default_agent = st.selectbox("Default Agent", [agent["name"] for agent in st.session_state.agents] if st.session_state.agents else ["No agents available"])
        max_reasoning_steps = st.number_input("Max Reasoning Steps", min_value=1, max_value=10, value=3)
    
    # Save orchestrator configuration
    if st.button("Set Orchestrator"):
        orchestrator_config = {
            "type": orchestrator_type
        }
        
        if orchestrator_type == "SimpleOrchestrator" or orchestrator_type == "ReactOrchestrator":
            if default_agent != "No agents available":
                orchestrator_config["default_agent"] = default_agent
        
        if orchestrator_type == "MultiAgentOrchestrator":
            if default_agent != "No agents available":
                orchestrator_config["default_agent"] = default_agent
                orchestrator_config["classifier"] = classifier_type
        
        if orchestrator_type == "BidOrchestrator":
            orchestrator_config["confidence_threshold"] = confidence_threshold
            orchestrator_config["team_mode"] = team_mode
        
        if orchestrator_type == "ReactOrchestrator":
            orchestrator_config["max_reasoning_steps"] = max_reasoning_steps
        
        st.session_state.orchestrator = orchestrator_config
        st.success(f"Orchestrator set to: {orchestrator_type}")
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 4: Generated Code
with tab4:
    st.header("Generate Code")
    
    # File name for the generated code
    file_name = st.text_input("File Name", "my_agent_system.py")
    
    # Generate code button
    if st.button("Generate Code"):
        if not st.session_state.agents:
            st.error("Please add at least one agent before generating code.")
        elif not st.session_state.orchestrator:
            st.error("Please select an orchestrator before generating code.")
        else:
            # Generate the Python code based on the configuration
            code = []
            
            # Add imports
            code.append('"""')
            code.append("Generated Moya agent system")
            code.append('"""')
            code.append("")
            code.append("import os")
            code.append("import sys")
            code.append("import dotenv")
            code.append("from moya.tools.tool_registry import ToolRegistry")
            code.append("from moya.registry.agent_registry import AgentRegistry")
            
            # Import orchestrator
            orchestrator_type = st.session_state.orchestrator["type"]
            code.append(f"from moya.orchestrators.{orchestrator_type.lower()} import {orchestrator_type}")
            
            # Import agent types
            agent_imports = set()
            for agent in st.session_state.agents:
                agent_imports.add(f"from moya.agents.{agent['agent_type'].lower()} import {agent['agent_type']}")
                if agent['agent_type'] != "OllamaAgent":  # OllamaAgent uses AgentConfig
                    agent_imports.add(f"from moya.agents.{agent['agent_type'].lower()} import {agent['agent_type']}Config")
            
            for imp in agent_imports:
                code.append(imp)
                
            # Import base agent config for OllamaAgent
            if any(agent['agent_type'] == "OllamaAgent" for agent in st.session_state.agents):
                code.append("from moya.agents.base_agent import AgentConfig")
            
            # Import tools
            if "memory_tools" in st.session_state.selected_tools:
                code.append("from moya.tools.ephemeral_memory import EphemeralMemory")
            if "consultation_tool" in st.session_state.selected_tools:
                code.append("from moya.tools.agent_consultation_tool import AgentConsultationTool")
            if "rag_tool" in st.session_state.selected_tools:
                code.append("from moya.tools.rag_search_tool import VectorSearchTool")
            if "web_search" in st.session_state.selected_tools:
                code.append("from moya.tools.search_tool import SearchTool")
            if "math_tool" in st.session_state.selected_tools:
                code.append("from moya.tools.math_tool import MathTool")
                
            code.append("")
            code.append("# Load environment variables")
            code.append("dotenv.load_dotenv()")
            code.append("")
            
            # Setup function
            code.append("def setup_system():")
            code.append('    """Set up agents and tools."""')
            code.append("    ")
            code.append("    # Create registries")
            code.append("    agent_registry = AgentRegistry()")
            code.append("    tool_registry = ToolRegistry()")
            code.append("    ")
            
            # Setup tools
            if "memory_tools" in st.session_state.selected_tools:
                code.append("    # Set up memory tools")
                code.append("    EphemeralMemory.configure_memory_tools(tool_registry)")
                code.append("    ")
                
            if "consultation_tool" in st.session_state.selected_tools:
                code.append("    # Set up consultation tools")
                code.append("    AgentConsultationTool.configure_consultation_tools(tool_registry, agent_registry)")
                code.append("    ")
                
            if "math_tool" in st.session_state.selected_tools:
                code.append("    # Set up math tools")
                code.append("    MathTool.configure_math_tools(tool_registry)")
                code.append("    ")
                
            # Setup agents
            for i, agent in enumerate(st.session_state.agents):
                code.append(f"    # {i+1}. Create {agent['name']} agent")
                if agent['agent_type'] == "OllamaAgent":
                    code.append(f"    {agent['name'].lower()}_config = AgentConfig(")
                else:
                    code.append(f"    {agent['name'].lower()}_config = {agent['agent_type']}Config(")
                
                code.append(f"        agent_name=\"{agent['name']}\",")
                code.append(f"        agent_type=\"ChatAgent\",")
                code.append(f"        description=\"{agent['description']}\",")
                code.append(f"        system_prompt=\"\"\"{agent['system_prompt']}\"\"\",")
                code.append("        tool_registry=tool_registry,")
                
                if agent['agent_type'] == "OllamaAgent":
                    code.append("        llm_config={")
                    code.append(f"            'model_name': \"{agent['model_name']}\",")
                    code.append(f"            'base_url': \"{agent['base_url']}\",")
                    code.append(f"            'temperature': {agent['temperature']}")
                    code.append("        }")
                elif agent['agent_type'] == "OpenAIAgent":
                    code.append(f"        model_name=\"{agent['model_name']}\",")
                    code.append(f"        temperature={agent['temperature']},")
                    code.append(f"        api_key=os.getenv(\"OPENAI_API_KEY\") or \"{agent['api_key']}\"")
                elif agent['agent_type'] == "AzureOpenAIAgent":
                    code.append(f"        model_name=\"{agent['model_name']}\",")
                    code.append(f"        temperature={agent['temperature']},")
                    code.append(f"        api_key=os.getenv(\"AZURE_OPENAI_API_KEY\") or \"{agent['api_key']}\",")
                    code.append(f"        api_base=\"{agent['api_base']}\",") 
                    code.append(f"        api_version=\"{agent['api_version']}\"")
                elif agent['agent_type'] == "BedrockAgent":
                    code.append(f"        model_id=\"{agent['model_name']}\",")
                    code.append(f"        temperature={agent['temperature']},")
                    code.append(f"        region=\"{agent['region']}\"")
                
                code.append("    )")
                code.append("    ")
                
                # Create agent instance
                code.append(f"    {agent['name'].lower()} = {agent['agent_type']}(agent_config={agent['name'].lower()}_config)")
                code.append(f"    agent_registry.register_agent({agent['name'].lower()})")
                code.append("    ")
            
            # Setup orchestrator
            code.append("    # Create orchestrator")
            
            if orchestrator_type == "SimpleOrchestrator":
                code.append("    orchestrator = SimpleOrchestrator(")
                code.append("        agent_registry=agent_registry,")
                if "default_agent" in st.session_state.orchestrator:
                    code.append(f"        default_agent_name=\"{st.session_state.orchestrator['default_agent']}\"")
                code.append("    )")
            
            elif orchestrator_type == "MultiAgentOrchestrator":
                code.append(f"    from moya.classifiers.{st.session_state.orchestrator['classifier'].lower()} import {st.session_state.orchestrator['classifier']}")
                code.append("    ")
                code.append(f"    classifier = {st.session_state.orchestrator['classifier']}(agent_registry=agent_registry)")
                code.append("    orchestrator = MultiAgentOrchestrator(")
                code.append("        agent_registry=agent_registry,")
                code.append("        classifier=classifier,")
                if "default_agent" in st.session_state.orchestrator:
                    code.append(f"        default_agent_name=\"{st.session_state.orchestrator['default_agent']}\"")
                code.append("    )")
            
            elif orchestrator_type == "BidOrchestrator":
                code.append("    orchestrator = BidOrchestrator(")
                code.append("        agent_registry=agent_registry,")
                code.append(f"        confidence_threshold={st.session_state.orchestrator['confidence_threshold']},")
                code.append(f"        team_mode={str(st.session_state.orchestrator['team_mode'])}")
                code.append("    )")
            
            elif orchestrator_type == "ReactOrchestrator":
                code.append("    orchestrator = ReactOrchestrator(")
                code.append("        agent_registry=agent_registry,")
                if "default_agent" in st.session_state.orchestrator:
                    code.append(f"        default_agent_name=\"{st.session_state.orchestrator['default_agent']}\",")
                code.append(f"        max_reasoning_steps={st.session_state.orchestrator['max_reasoning_steps']}")
                code.append("    )")
            
            code.append("    ")
            code.append("    return orchestrator")
            code.append("")
            
            # Main function
            code.append("def main():")
            code.append('    """Run the interactive agent system."""')
            code.append("    print(\"Starting agent system...\")")
            code.append("    print(\"\\nInitializing system...\")")
            code.append("    ")
            code.append("    try:")
            code.append("        orchestrator = setup_system()")
            code.append("        print(\"System initialized successfully!\")")
            code.append("    except Exception as e:")
            code.append("        print(f\"Error setting up system: {str(e)}\")")
            code.append("        return")
            code.append("    ")
            code.append("    # For conversation history")
            code.append("    thread_id = \"conversation_thread\"")
            code.append("    print(\"\\nWelcome to your Moya Agent System\")")
            code.append("    print(\"Type 'exit' to quit.\\n\")")
            code.append("    ")
            code.append("    def stream_callback(chunk):")
            code.append("        \"\"\"Print response chunks as they arrive.\"\"\"")
            code.append("        print(chunk, end=\"\", flush=True)")
            code.append("    ")
            code.append("    while True:")
            code.append("        user_input = input(\"\\nYou: \")")
            code.append("        if user_input.lower() == \"exit\":")
            code.append("            print(\"\\nGoodbye!\")")
            code.append("            break")
            code.append("        ")
            code.append("        # Store user message")
            code.append("        EphemeralMemory.store_message(")
            code.append("            thread_id=thread_id,")
            code.append("            sender=\"user\",")
            code.append("            content=user_input")
            code.append("        )")
            code.append("        ")
            code.append("        # Process with orchestrator")
            code.append("        print(\"\\nAssistant: \", end=\"\", flush=True)")
            code.append("        response = orchestrator.orchestrate(")
            code.append("            thread_id=thread_id,")
            code.append("            user_message=user_input,")
            code.append("            stream_callback=stream_callback")
            code.append("        )")
            code.append("        ")
            code.append("        # Store assistant response")
            code.append("        EphemeralMemory.store_message(")
            code.append("            thread_id=thread_id,")
            code.append("            sender=\"assistant\",")
            code.append("            content=response")
            code.append("        )")
            code.append("")
            code.append("if __name__ == \"__main__\":")
            code.append("    main()")
            
            # Display the generated code
            st.code("\n".join(code), language="python")
            
            # Save to file option
            if st.button("Save to File"):
                try:
                    file_path = Path(file_name)
                    with open(file_path, 'w') as f:
                        f.write("\n".join(code))
                    st.success(f"Code saved to {file_path.absolute()}")
                except Exception as e:
                    st.error(f"Error saving file: {str(e)}")
            
            st.session_state.code_generated = True

# Add a reset button at the bottom
if st.button("Reset All"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# Display instructions
with st.expander("How to Use"):
    st.markdown("""
    ### Using the Moya Agent Builder
    
    1. **Add Agents**: Use the sidebar to configure and add agents to your system.
    2. **Select Tools**: Choose which tools your agents will have access to.
    3. **Choose Orchestrator**: Select and configure your orchestration strategy.
    4. **Generate Code**: Preview the generated code and save it to a file.
    5. **Run Your System**: Execute the saved Python file to run your custom agent system.
    
    ### Requirements
    
    - The Moya library must be installed (`pip install moya-ai`).
    - For Ollama agents, ensure Ollama is running locally.
    - For OpenAI agents, you'll need an API key.
    """)