import ollama

def chat_with_model():
    model_name = "deepseek-r1"
    system_message = {
        "role": "system",
        "content": "You are a helpful travel assistant that provides flight details, travel itineraries, famous places to visit, and useful tips."
    }
    
    print("\nType 'exit' to quit the chat.\n")
    
    Chat_History = []

    while True:
        try:
            user_prompt = input("You: ")
            if user_prompt.lower() == "exit":
                print("Goodbye!")
                break

            response = ollama.chat(
                model=model_name,
                messages=[system_message, {"role": "user", "content": user_prompt}],
                stream=True,
                options={
                    "temperature": 0.5,
                    "max_tokens": 300,
                    "top_p": 0.8,
                    "top_k": 40,
                    "stop": ["User:", "Assistant:"],
                }
            )
            
            print("\nAssistant:", end=" ")
            for chunk in response:
                if "message" in chunk and "content" in chunk["message"]:
                    print(chunk["message"]["content"], end='', flush=True)
            print("\n")
            
            Chat_History.append({"role": "user", "content": user_prompt})
            Chat_History.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nChat terminated. Goodbye!")
            break

if __name__ == "__main__":
    chat_with_model()