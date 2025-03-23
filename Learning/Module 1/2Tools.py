from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage

llm = ChatOllama(model="llama3.1:8b", device="cuda")

def multiply(a:int, b:int)->int:
    '''
    multiply a and b 
    Args:
        a: int
        b: int
    '''
    return a*b

llm_with_tools = llm.bind_tools([multiply])

# for chunk in llm_with_tools.stream([HumanMessage(content="What is 12 multipled by 4", name="Shubham")]):
#     print(chunk.text(),end="",flush=True)

tool_call = llm_with_tools.invoke([HumanMessage(content="What is 12 multipled by 4", name="Shubham")])
print(tool_call.text())
print(tool_call.tool_calls[0])