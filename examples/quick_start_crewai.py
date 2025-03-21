"""
Interactive chat example using OpenAI agent with conversation memory.
"""

import os
from moya.memory.in_memory_repository import InMemoryRepository
from moya.tools.tool_registry import ToolRegistry
from moya.tools.memory_tool import MemoryTool
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.simple_orchestrator import SimpleOrchestrator
from moya.agents.crewai_agent import CrewAIAgent, CrewAIAgentConfig


def setup_agent():
    # Set up memory components
    memory_repo = InMemoryRepository()
    memory_tool = MemoryTool(memory_repository=memory_repo)
    tool_registry = ToolRegistry()
    tool_registry.register_tool(memory_tool)

    # Create OpenAI agent with memory capabilities
    agent = CrewAIAgent(
        agent_name="eli5_agent",
        description="An intelligent agent that can explain things to a five year old.",
        agent_config=CrewAIAgentConfig(system_prompt="Can you explain this to me like I'm five?",
                                       api_key=os.environ.get("OPENAI_API_KEY"),
                                       model_name="gpt-4o"
                                       ),
        tool_registry=tool_registry
    )
    agent.setup()

    # Set up registry and orchestrator
    agent_registry = AgentRegistry()
    agent_registry.register_agent(agent)
    orchestrator = SimpleOrchestrator(
        agent_registry=agent_registry,
        default_agent_name="eli5_agent"
    )

    return orchestrator, agent


def format_conversation_context(messages):
    context = "\nPrevious conversation:\n"
    for msg in messages:
        # Access Message object attributes properly using dot notation
        sender = "User" if msg.sender == "user" else "Assistant"
        context += f"{sender}: {msg.content}\n"
    return context


def main():
    orchestrator, agent = setup_agent()
    thread_id = "interactive_chat_001"

    print("Welcome to Interactive Chat! (Type 'quit' or 'exit' to end)")
    print("-" * 50)

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for exit command
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break

        # Store the user message first
        agent.call_tool(
            tool_name="MemoryTool",
            method_name="store_message",
            thread_id=thread_id,
            sender="user",
            content=user_input
        )

        # Get conversation context
        previous_messages = agent.get_last_n_messages(thread_id, n=5)

        # Add context to the user's message if there are previous messages
        if previous_messages:
            context = format_conversation_context(previous_messages)
            enhanced_input = f"{context}\nCurrent user message: {user_input}"
        else:
            enhanced_input = user_input

        # Print Assistant prompt
        print("\nAssistant: ", end="", flush=True)

        # Define callback for streaming
        def stream_callback(chunk):
            print(chunk, end="", flush=True)

        # Get response using stream_callback
        response = orchestrator.orchestrate(
            thread_id=thread_id,
            user_message=enhanced_input,
            stream_callback=stream_callback
        )

        # Print newline after response
        print()

        # Store the assistant's response
        agent.call_tool(
            tool_name="MemoryTool",
            method_name="store_message",
            thread_id=thread_id,
            sender="assistant",
            content=response
        )


if __name__ == "__main__":
    main()
