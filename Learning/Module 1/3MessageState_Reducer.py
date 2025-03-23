from IPython.display import Image, display
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END,MessagesState
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}



llm = ChatOllama(
    model="llama3.1:8b",
    device="cuda"
    )


def multiply(a:int, b:int)->int:
    '''
    multiply a and b 
    Args:
        a: int
        b: int
    '''
    return a*b

llm_with_tools = llm.bind_tools([multiply])

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()

# View
Image(graph.get_graph(xray=True).draw_mermaid_png(output_file_path=r"Learning\Module 1\3MessageState.png"))

messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
for m in messages['messages']:
    m.pretty_print()

messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3")})
for m in messages['messages']:
    m.pretty_print()

