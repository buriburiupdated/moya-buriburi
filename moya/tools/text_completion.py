import sys
from typing import Dict, Any
from moya.agents.ollama_agent import OllamaAgent
from moya.agents.base_agent import AgentConfig
from moya.tools.tool_registry import ToolRegistry
from moya.tools.base_tool import BaseTool
import re

class TextAutocomplete:
    """Tools for text autocompletion capabilities."""

    _agent = None
    _last_input = None
    _last_output = None

    @staticmethod
    def _get_agent():
        if TextAutocomplete._agent is None:
            tool_registry = ToolRegistry()
            agent_config = AgentConfig(
                agent_name="text_completer",
                agent_type="CompletionAgent",
                description="A human-like text completion assistant",
                system_prompt="""
                You are a text auto-complete bot. Your duty is to recommend phrases to the user to help them complete the sentence.
                Please stick to the context and only help in TEXT COMPLETION. Predict at most the next 5-6 words, 
                ensuring the completion is grammatically correct and contextually appropriate. You do not have to andswer questions asked by the user.
                """,
                tool_registry=tool_registry,
                llm_config={
                    'model_name': "qwen:14b",
                    'temperature': 0.3,  # Lowered for more controlled completions
                    'base_url': "http://localhost:11434",
                    'context_window': 1024
                }
            )
            TextAutocomplete._agent = OllamaAgent(agent_config)
            test_response = TextAutocomplete._agent.handle_message("test")
            if not test_response:
                raise Exception("No response from Ollama")
        return TextAutocomplete._agent

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("User:", "").strip().rstrip(" .")
        return text

    @staticmethod
    def _clean_completion(completion: str) -> str:
        completion = completion.strip()
        completion = re.sub(r'\b\w{1,2}[.?!]*$', '', completion)  # Remove incomplete words
        return completion.strip()

    @staticmethod
    def complete_text(text: str) -> str:
        try:
            if text == TextAutocomplete._last_input:
                return TextAutocomplete._last_output

            agent = TextAutocomplete._get_agent()
            text = TextAutocomplete._clean_text(text)
            if not text:
                return ""

            response = agent.handle_message(text)
            if not response:
                return ""

            completion = " ".join(response.strip().split()[:6])  # Limit completion to 5-6 words
            completion = TextAutocomplete._clean_completion(completion)

            TextAutocomplete._last_input = text
            TextAutocomplete._last_output = completion

            return completion

        except Exception as e:
            print(f"\nError getting completion: {e}")
            return ""

    @staticmethod
    def configure_autocomplete_tools(tool_registry: ToolRegistry) -> None:
        tool_registry.register_tool(
            BaseTool(
                name="TextCompletionTool",
                function=TextAutocomplete.complete_text,
                description="Complete text using AI model",
                parameters={
                    "text": {
                        "type": "string",
                        "description": "The text to be completed",
                        "required": True
                    }
                },
                return_type="string",
                return_description="Completed text suggestion"
            )
        )
