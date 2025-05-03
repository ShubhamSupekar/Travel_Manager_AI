import os
from langchain.chat_models import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")

# convert_pydantic_to_openai_function(Tagging)

tagging_functions = [convert_pydantic_to_openai_function(Tagging)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=50,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

model_with_functions = model.bind(
    functions=tagging_functions,
    function_call={"name": "Tagging"}
)

tagging_chain = prompt | model_with_functions

print(tagging_chain.invoke({"input": "I love langchain"}))
print()
print(tagging_chain.invoke({"input": "non mi piace questo cibo"})) # This means "I don't like this food" in Italian
# O/P:
# content='' additional_kwargs={'function_call': {'arguments': '{"sentiment":"pos","language":"en"}', 'name': 'Tagging'}} response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 108, 'total_tokens': 119, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-fcba278a-cd3a-4c26-941c-64ba96688c15-0'
# content='' additional_kwargs={'function_call': {'arguments': '{"sentiment":"neg","language":"it"}', 'name': 'Tagging'}} response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 111, 'total_tokens': 122, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-e8f2eee1-3dce-47d0-8578-84cd7cc25db7-0'

# now parsing the output in structured format
tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()
print()
print(tagging_chain.invoke({"input": "non mi piace questo cibo"}))
# O/P:
# {'sentiment': 'neg', 'language': 'it'}