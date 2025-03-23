import json
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

# Load the LLM
llm = ChatOllama(model="llama3.1:8b", device="cuda")

# Define Tools
@tool
def multiply(a: int, b: int) -> str:
    """Multiplies two numbers and returns the result as a sentence."""
    return f"The result of {a} multiplied by {b} is {a * b}."

@tool
def weather(city: str) -> str:
    """Fetches the weather for a given city."""
    return f"The current weather in {city} is 25°C and sunny."

@tool
def summarize(text: str) -> str:
    """Summarizes a given text."""
    return f"Summary: {text[:50]}..."

# Bind tools to LLM
llm_with_tools = llm.bind_tools([multiply, weather, summarize])

def process_query(query):
    """
    Processes the user query with tool calls and ensures the final response is well-phrased.
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

    if tool_calls:
        tool_responses = {}
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            if tool_name == "multiply":
                tool_responses[tool_name] = multiply.invoke(tool_args)
            elif tool_name == "weather":
                tool_responses[tool_name] = weather.invoke(tool_args)
            elif tool_name == "summarize":
                tool_responses[tool_name] = summarize.invoke(tool_args)

        # Convert tool responses into readable sentences
        formatted_response = " ".join(tool_responses.values())

        print("\nAI:", formatted_response if formatted_response else "I couldn't generate a response.")
    else:
        print()

# Example Queries
queries = [
    "What is 12 multiplied by 4?",
    "Tell me the weather in New York.",
    "Summarize this article: LangChain is an AI framework for building applications...",
    "Who is the president of the USA?"  # Likely no tool call here
]

for query in queries:
    print("\nUser:", query)
    process_query(query)
