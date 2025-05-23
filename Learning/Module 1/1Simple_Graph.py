from typing import TypedDict

class State(TypedDict):
    graph_state: str

def node_1(state):
    print("---Node 1---")
    return{"graph_state": state['graph_state']+" I am"}

def node_2(state):
    print("---Node 2---")
    return{"graph_state": state['graph_state']+" happy!"}

def node_3(state):
    print("---Node 3---")
    return{"graph_state": state['graph_state']+" sad!"}

import random
from typing import Literal

def decide_node(state) -> Literal["node_2","node_3"]:

    user_input = state['graph_state'] 

    if random.random() < 0.5:
        # 50% chance of time we are returning node_2
        return "node_2"
    
    # 50% chance of time we are returning node_3
    return "node_3"

from langgraph.graph import StateGraph,START,END
from IPython.display import Image, display

# Build graph

# adding state
builder = StateGraph(State)

# adding nodes
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# addign logic to connect the nodes with edges 
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_node)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph = builder.compile()

Image(graph.get_graph(xray=True).draw_mermaid_png(output_file_path="Learning\Module 1\1simple_graph.png"))

print(graph.invoke({"graph_state" : "Hi, this is Lance."}))