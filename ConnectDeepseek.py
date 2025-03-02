import ollama
import json
import os

HISTORY_FILE = "History.json"  

def load_chat_history():
    # loding hestory json file 
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []  
    return []


def save_chat_history(chat_history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(chat_history, file, indent=4, ensure_ascii=False)


def chat_with_model():

    model_name = "deepseek-r1"

    chat_history = load_chat_history()

    if not chat_history:
        chat_history.append({"role": "system", "content": "You are a helpful travel assistant that provides flight details, travel itineraries, famous places to visit, and useful tips."})

    print("\nType 'exit' to quit the chat.\n")

    while True:
        try:
            user_prompt = input("You: ")
            if user_prompt.lower() == "exit":
                print("Saving chat history... Goodbye!")
                save_chat_history(chat_history)  
                break

            chat_history.append({"role": "user", "content": user_prompt})

            response_stream = ollama.chat(
                model=model_name,
                messages=chat_history,  
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
            assistant_response = ""

            for chunk in response_stream:
                if "message" in chunk and "content" in chunk["message"]:
                    text = chunk["message"]["content"]
                    print(text, end='', flush=True)
                    assistant_response += text
            print("\n")

            chat_history.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\nSaving chat history... Goodbye!")
            save_chat_history(chat_history)
            break

if __name__ == "__main__":
    chat_with_model()
