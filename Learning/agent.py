import json
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

# Load the LLM
llm = ChatOllama(model="llama3.1:8b", device="cuda")

# Define Tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers and returns the result."""
    return a * b

@tool
def weather(city: str) -> str:
    """Fetches the weather for a given city."""
    return f"The weather in {city} is 25°C and sunny."

@tool
def summarize(text: str) -> str:
    """Summarizes a given text."""
    return f"Summary: {text[:50]}..."

# Bind tools to LLM
llm_with_tools = llm.bind_tools([multiply, weather, summarize])

def process_query(query):
    """
    Processes the user query by calling tools only when necessary.
    """
    # Invoke LLM to check if it needs a tool
    response_stream = llm_with_tools.stream(query)
    tool_calls = []
    initial_output = ""

    for chunk in response_stream:
        if chunk.tool_calls:
            tool_calls.extend(chunk.tool_calls)
        if chunk.content:
            initial_output += chunk.content
            print(chunk.content, end="", flush=True)

    # Only proceed with relevant tool calls
    if tool_calls:
        tool_responses = {}

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Only execute tools that match the original query intent
            if "multiply" in query.lower() and tool_name == "multiply":
                tool_responses[tool_name] = multiply.invoke(tool_args)
            elif "weather" in query.lower() and tool_name == "weather":
                tool_responses[tool_name] = weather.invoke(tool_args)
            elif "summarize" in query.lower() and tool_name == "summarize":
                tool_responses[tool_name] = summarize.invoke(tool_args)
        
        # If no valid tool calls were found, skip processing
        if not tool_responses:
            print("\nAI:", initial_output)
            return

        # Format tool outputs into a readable message
        formatted_response = json.dumps(tool_responses, indent=2)

        # Send tool responses back to LLM to generate a natural response
        final_response = llm.invoke(
            f"Based on the following tool outputs, generate a natural-sounding response:\n\n{formatted_response}"
        )

        print("\nAI:", final_response)
    else:
        print("\nAI:", initial_output)

# Example Queries
queries = [
    "What is 12 multiplied by 4?",
    "Tell me the weather in New York.",
    "Summarize this article: LangChain is an AI framework for building applications...",
    "Who is the president of the USA?"  # Should NOT call any tools
]

for query in queries:
    print("\nUser:", query)
    process_query(query)
