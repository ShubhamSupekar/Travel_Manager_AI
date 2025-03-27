from langchain_ollama import ChatOllama
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import re

# initialize the model
llm = ChatOllama(
    model="llama3.1:8b",
    device="cuda"
)

# defining function
def multiply_numbers(*args):
    print(f"Received args: {args}")  # Debugging output
    
    # Extract the first argument from the tuple, assuming it contains the input string.
    if not args:
        print("Error: No input provided.")
        return None
    input_string = args[0]
    
    # Finds all numbers in the string
    matches = re.findall(r"(\d+)", input_string)

    if len(matches) == 2:
        a, b = map(int, matches)  # Convert to integers
        print(f"a = {a}, b = {b}")  # Output: a = 6, b = 7
        return a * b
    else:
        print("Error: Could not extract two numbers.")
        return None

# defining tool
multiply_tool = Tool(
    name="multiply",
    description="Multiply two numbers. Provide 'a' and 'b' as input parameters.",
    func=multiply_numbers
)

# initialize the agent
agent = initialize_agent(
    tools=[multiply_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.invoke("What is 6 multiplied by 7?")
print(response)
