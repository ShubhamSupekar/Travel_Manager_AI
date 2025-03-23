from IPython.display import Image, display
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END,MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


def multiply(a:int, b:int)->int:
    '''
    multiply a and b 
    Args:
        a: int
        b: int
    '''
    return a*b


llm = ChatOllama(model="llama3.1:8b",device="cuda")

llm_with_tools = llm.bind_tools([multiply])

def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools",ToolNode([multiply]))
builder.add_edge(START,"tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)

builder.add_edge("tools", END)
graph = builder.compile()


# View
Image(graph.get_graph(xray=True).draw_mermaid_png(output_file_path=r"Learning\Module 1\4router.png"))

from langchain_core.messages import HumanMessage
messages = [HumanMessage(content="Hello world")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()

messages = [HumanMessage(content="Hello, what is 11 multiplied by 2?")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()