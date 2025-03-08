from langchain_ollama import ChatOllama
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory

def chat_with_model():
    # Initialize the LLM with your chosen model
    llm = ChatOllama(model="deepseek-r1", temperature=0.5)
    
    # Initialize ConversationBufferMemory to store chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Define the system message (the chatbot's instructions)
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a helpful travel assistant that provides flight details, travel itineraries, famous places to visit, and useful travel tips."
    )
    
    # Define the human message template (for user queries)
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    
    # Combine the system and human messages into a chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    
    # Create an LLMChain with the prompt and memory.
    chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory)
    
    print("Type 'exit' to end the conversation.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Run the chain with the current input. The memory is automatically updated.
        response = chain.run(input=user_input)
        print("\nAssistant:", response, "\n")

if __name__ == "__main__":
    chat_with_model()
