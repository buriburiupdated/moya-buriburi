"""
Generated Moya agent system
"""

import os
import sys
import dotenv
from moya.tools.tool_registry import ToolRegistry
from moya.registry.agent_registry import AgentRegistry
from moya.orchestrators.multi_agent_orchestrator import MultiAgentOrchestrator
from moya.agents.ollama_agent import OllamaAgent
from moya.agents.base_agent import AgentConfig
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.tools.math_tool import MathTool

# Load environment variables
dotenv.load_dotenv()

def setup_system():
    """Set up agents and tools."""

    # Create registries
    agent_registry = AgentRegistry()
    tool_registry = ToolRegistry()

    # Set up memory tools
    EphemeralMemory.configure_memory_tools(tool_registry)

    # Set up math tools
    MathTool.configure_math_tools(tool_registry)

    # 1. Create Math Agent agent
    mathagent_config = AgentConfig(
        agent_name="Math Agent",
        agent_type="ChatAgent",
        description="This agent can solve math problems",
        system_prompt="""You are a math specialist. You excel at solving mathematical problems, 
        from basic arithmetic to advanced calculus and statistics.

        Provide step-by-step solutions when appropriate, and explain your reasoning clearly.
        Always double-check your calculations.

        When consulted by other agents, focus on giving precise and accurate mathematical explanations.
        Be thorough but concise in your responses.""",
        tool_registry=tool_registry,
        llm_config={
            'model_name': "llama3.1:latest",
            'base_url': "http://localhost:11434",
            'temperature': 0.7
        }
    )

    mathagent = OllamaAgent(agent_config=mathagent_config)
    agent_registry.register_agent(mathagent)

    # 2. Create Code Agent agent
    codeagent_config = AgentConfig(
        agent_name="Code Agent",
        agent_type="ChatAgent",
        description="A specialized agent for programming and software development",
        system_prompt="""You are a coding specialist. You excel at programming, software development,
        and technical problem-solving.

        When providing code examples, ensure they are correct, efficient, and well-commented.
        Consider best practices and edge cases in your solutions.

        When consulted by other agents, focus on practical implementations and clear explanations
        of coding concepts and techniques.""",
        tool_registry=tool_registry,
        llm_config={
            'model_name': "codellama",
            'base_url': "http://localhost:11434",
            'temperature': 0.7
        }
    )

    codeagent = OllamaAgent(agent_config=codeagent_config)
    agent_registry.register_agent(codeagent)

    # Create orchestrator
    from moya.classifiers.llm_classifier import LLMClassifier

    classifier = LLMClassifier(
        llm_agent=mathagent,
        default_agent="Math Agent"
    )
    orchestrator = MultiAgentOrchestrator(
        agent_registry=agent_registry,
        classifier=classifier,
        default_agent_name="Math Agent"
    )

    return orchestrator

def main():
    """Run the interactive agent system."""
    print("Starting agent system...")
    print("\nInitializing system...")

    try:
        orchestrator = setup_system()
        print("System initialized successfully!")
    except Exception as e:
        print(f"Error setting up system: {str(e)}")
        return

    # For conversation history
    thread_id = "conversation_thread"
    print("\nWelcome to your Moya Agent System")
    print("Type 'exit' to quit.\n")

    def stream_callback(chunk):
        """Print response chunks as they arrive."""
        print(chunk, end="", flush=True)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break

        # Store user message
        EphemeralMemory.store_message(
            thread_id=thread_id,
            sender="user",
            content=user_input
        )

        # Process with orchestrator
        print("\nAssistant: ", end="", flush=True)
        response = orchestrator.orchestrate(
            thread_id=thread_id,
            user_message=user_input,
            stream_callback=stream_callback
        )

        # Store assistant response
        EphemeralMemory.store_message(
            thread_id=thread_id,
            sender="assistant",
            content=response
        )

if __name__ == "__main__":
    main()