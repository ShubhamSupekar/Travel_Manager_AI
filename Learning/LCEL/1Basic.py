# Langchain expression language 
import os
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=50,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
) 

output_parser = StrOutputParser()

chain = prompt | model | output_parser

print(chain.invoke({"topic":"chicken"}))
