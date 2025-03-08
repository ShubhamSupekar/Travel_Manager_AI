from langchain_ollama import ChatOllama


def chat_with_model():
    llm = ChatOllama(
        model="deepseek-r1", 
        temperature=0.5
        )

    messages = [
        ("system", "You are a helpful travel assistant that provides flight details, travel itineraries, famous places to visit, and useful tips.")
    ]

    while True:
        input_prompt = str(input("You: "))

        if input_prompt.lower() == "exit":
            print("Goodbye!")
            break

        messages.append(("user", input_prompt))

        print("\nAssistant:", end=" ")
        assistant_response = ""
        for chunk in llm.stream(messages):
            print(chunk.text(),end="",flush=True)
            assistant_response += chunk.text()

        messages.append(("assistant", assistant_response))
        print("\n")

if __name__ == "__main__":
    chat_with_model()