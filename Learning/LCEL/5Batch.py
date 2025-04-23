import os 
import langchain.prompts
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
import langchain

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=50,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

output_parser = StrOutputParser()

prompt = langchain.prompts.ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}",
)

chain = prompt | model | output_parser

print(chain.batch([{"topic": "cats"}, {"topic": "dogs"}]))