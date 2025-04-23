import os
from langchain.llms import OpenAI
import json
from langchain_openai import ChatOpenAI

simple_model = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0,
    max_tokens=50,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
simple_chain = simple_model | json.loads

challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"

# print(simple_model.invoke(challenge))

# print(simple_chain.invoke(challenge)) this line will give the error beacuse model output is not a valid json

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=50,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# if we get failed to get the response from the simple model then fall back to the new model
final_chain = simple_model.with_fallbacks([model])

print(final_chain.invoke(challenge))