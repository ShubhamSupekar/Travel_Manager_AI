from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os

# Set OpenAI API Key
# os.environ["OPENAI_API_KEY"] = "your-api-key"
api_key = os.getenv("OPENAI_API_KEY") 

# Initialize the OpenAI model with cost-optimized settings
chat = ChatOpenAI(
    model="gpt-3.5-turbo",      # Using GPT-4o
    temperature=0.2,     # Low temperature for concise responses
    max_tokens=50,       # Limit response length to save cost
    request_timeout=10,   # Timeout to avoid unnecessary API calls
    openai_api_key=api_key # API key
)

try:
    # Send a simple test message
    response = chat([HumanMessage(content="Hello!")])

    print("✅ API Key is working! Response received:")
    print(response.content)

except Exception as e:
    print(f"⚠️ Error: {e}")

