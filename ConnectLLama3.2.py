from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

def chat_with_model():
    llm = ChatOllama(
        model="llama3.2"
        )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful travel assistant that provides flight details, travel itineraries, famous places to visit, and useful tips."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")  
    ])


    while True:
        input_prompt = str(input("You: "))

        if input_prompt.lower() == "exit":
            print("Goodbye!")
            break

        chat_history = memory.load_memory_variables({})["chat_history"]
        
        formatted_prompt = prompt.format_messages(chat_history=chat_history, input=input_prompt)

        print("\nAssistant:", end=" ")
        assistant_response = ""
        for chunk in llm.stream(formatted_prompt):
            print(chunk.text(),end="",flush=True)
            assistant_response += chunk.text()

        memory.save_context({"input": input_prompt}, {"output": assistant_response})

        print("\n")

if __name__ == "__main__":
    chat_with_model()