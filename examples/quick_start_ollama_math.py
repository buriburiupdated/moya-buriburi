"""
Interactive RAG demo that vectorizes docs folder HTML files and uses Ollama to answer questions
about the Moya framework with proper document citations.
"""

import os
import sys
import glob

from langchain_ollama import OllamaEmbeddings

from moya.agents.ollama_agent import OllamaAgent
from moya.agents.base_agent import AgentConfig
from moya.tools.tool_registry import ToolRegistry
from moya.tools.math_tool import MathTool
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.conversation.message import Message
from moya.conversation.thread import Thread


def setup_agent(vector_store):
    """Set up the Ollama agent with RAG search tool."""
    # Set up the tool registry and configure tools
    tool_registry = ToolRegistry()
    MathTool.configure_math_tools(tool_registry)
    EphemeralMemory.configure_memory_tools(tool_registry)
    
    # Create agent configuration for Ollama
    system_prompt = """You are a knowledgeable assistant that can solve 
    When answering questions about Moya, use the VectorSearchTool to search the documentation.
    Always cite sources by mentioning which file the information comes from.
    Be concise and helpful."""
    
    agent_config = AgentConfig(
        agent_name="moya_docs_assistant",
        agent_type="ChatAgent",
        description="Ollama agent integrated with RAG search of Moya documentation",
        system_prompt=system_prompt,
        tool_registry=tool_registry,
        llm_config={
            "model_name": "llama3.1:latest",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
            "context_window": 4096
        }
    )
    
    # Instantiate the Ollama agent
    agent = OllamaAgent(agent_config)
    
    # Verify connection
    try:
        test_response = agent.handle_message("test connection")
        if not test_response:
            raise Exception("No response from Ollama test query")
    except Exception as e:
        print("\nError: Make sure Ollama is running and the model is downloaded:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull model: ollama pull llama3.1:latest")
        sys.exit(1)
    
    # Set up agent registry and orchestrator
    agent_registry = AgentRegistry()
    agent_registry.register_agent(agent)
    orchestrator = SimpleOrchestrator(
        agent_registry=agent_registry,
        default_agent_name="moya_docs_assistant"
    )
    
    return orchestrator, agent


def format_conversation_context(messages) -> str:
    context = "\nPrevious conversation:\n"
    for msg in messages:
        sender = "User" if msg.sender == "user" else "Assistant"
        context += f"{sender}: {msg.content}\n"
    return context


def main():
    docs_dir = "../docs"
    path = "faiss-index"
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = FAISSCPUVectorstoreRepository(path, embeddings)
    vector_store.create_vectorstore()

    for file in glob.glob(f"{docs_dir}/*"):
        vector_store.load_file(file)
    
    # Set up agent with RAG search tool
    orchestrator, agent = setup_agent(vector_store)
    thread_id = "moya_docs_thread"
    
    # Initialize conversation memory
    EphemeralMemory.memory_repository.create_thread(Thread(thread_id=thread_id))
    
    print("\n" + "=" * 80)
    print("Welcome to the Moya Documentation Assistant!")
    print("Ask questions about the Moya framework and get answers from the documentation.")
    print("Type 'quit' or 'exit' to end the session.")
    print("=" * 80)
    
    # Example questions to help users get started
    print("\nExample questions you can ask:")
    print("- What is Moya?")
    print("- How do I create an agent?")
    print("- What types of agents are supported?")
    print("- How does memory management work?")
    print("- What is tool registry?")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
        
        # Store user message
        EphemeralMemory.memory_repository.append_message(
            thread_id, 
            Message(thread_id=thread_id, sender="user", content=user_input)
        )
        
        # Retrieve conversation context
        thread = EphemeralMemory.memory_repository.get_thread(thread_id)
        previous_messages = thread.get_last_n_messages(n=5)
        context = format_conversation_context(previous_messages) if previous_messages else ""
        
        # Construct prompt that includes an instruction to use RAG
        enhanced_input = (
            f"{context}"
            f"User: {user_input}\n\n"
            f"Remember to use the VectorSearchTool to search for information in the documentation. "
            f"Search query: {user_input}"
        )
        
        print("\nAssistant: ", end="", flush=True)
        
        try:
            # Try streaming response first
            response = ""
            try:
                for chunk in agent.handle_message_stream(enhanced_input):
                    if chunk:
                        print(chunk, end="", flush=True)
                        response += chunk
            except Exception as e:
                # Fall back to non-streaming
                response = agent.handle_message(enhanced_input)
                print(response)
            
            print()  # Add newline after response
            
            # Store agent response in memory
            EphemeralMemory.memory_repository.append_message(
                thread_id,
                Message(thread_id=thread_id, sender="assistant", content=response)
            )
            
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            continue


if __name__ == "__main__":
    main()