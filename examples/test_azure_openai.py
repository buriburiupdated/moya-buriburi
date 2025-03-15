"""
Interactive Azure OpenAI chat interface.
"""

import os
import sys
import dotenv
from openai import AzureOpenAI

dotenv.load_dotenv()

def create_azure_client():
    """Create and return an Azure OpenAI client."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    print("\nChecking Azure OpenAI configuration:")
    print(f"API Base: {api_base}")
    print(f"API Version: {api_version}")
    print(f"API Key: {'Set' if api_key else 'Not Set'}")
    print(f"Model Name: gpt-4o")
    
    # Verify credentials are set
    if not api_key or not api_base:
        print("Error: Azure OpenAI credentials not set. Please set environment variables:")
        print("- AZURE_OPENAI_API_KEY")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_VERSION (optional)")
        sys.exit(1)
        
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=api_base
    )

def interactive_chat():
    """Run an interactive chat session with Azure OpenAI."""
    client = create_azure_client()
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Respond with detailed but concise answers."}
    ]
    
    print("\nWelcome to Azure GPT Chat!")
    print("Type your messages and press Enter. Type 'quit' or 'exit' to end the conversation.")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ["quit", "exit"]:
            print("\nGoodbye!")
            break
            
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        try:
            print("\nAssistant: ", end="", flush=True)
            
            # Try non-streaming first
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    stream=False
                )
                content = response.choices[0].message.content
                print(content)
                messages.append({"role": "assistant", "content": content})
                
            except Exception as e:
                print(f"\nFalling back to streaming due to: {str(e)}")
                
                # Fall back to streaming with better error handling
                stream = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    stream=True
                )
                
                response_content = ""
                try:
                    for chunk in stream:
                        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                            if chunk.choices[0].delta.content is not None:
                                content = chunk.choices[0].delta.content
                                print(content, end="", flush=True)
                                response_content += content
                                
                    # Add assistant response to history
                    if response_content:
                        messages.append({"role": "assistant", "content": response_content})
                    print()  # New line after response
                except Exception as stream_error:
                    print(f"\nError during streaming: {str(stream_error)}")
            
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {str(e)}")
            print("Please check your Azure OpenAI configuration.")

if __name__ == "__main__":
    interactive_chat()