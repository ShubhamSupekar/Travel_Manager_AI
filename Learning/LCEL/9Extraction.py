import os
from langchain.chat_models import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from typing import Optional
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser


class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")


class Information(BaseModel):
    """Information to extract."""
    people: List[Person] = Field(description="List of info about people")


model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=50,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

extraction_functions = [convert_pydantic_to_openai_function(Information)]
extraction_model = model.bind(functions=extraction_functions, function_call={"name": "Information"})    

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])

extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()
print(extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"}))
print()
# O/P:
# {'people': [{'name': 'Joe', 'age': 30}, {'name': 'Martha', 'age': None}]}

# extraction with key name
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="people")
print(extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"}))
# O/P:
# [{'name': 'Joe', 'age': 30}, {'name': 'Martha', 'age': None}]