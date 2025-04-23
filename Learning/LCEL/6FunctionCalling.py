import os
from langchain.chat_models import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")

weather_function = convert_pydantic_to_openai_function(WeatherSearch)


model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=50,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

print(model.invoke("what is the weather in SF today?", functions=[weather_function]))

model_with_function = model.bind(functions=[weather_function])

print(model_with_function.invoke("what is the weather in sf?"))
# O/P:
# content='' additional_kwargs={'function_call': {'arguments': '{"airport_code":"SFO"}', 'name': 'WeatherSearch'}} response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 70, 'total_tokens': 88, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'function_call', 'logprobs': None} id='run-70075f6f-5015-499e-acc9-746365243449-0'
# content='' additional_kwargs={'function_call': {'arguments': '{"airport_code":"SFO"}', 'name': 'WeatherSearch'}} response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 69, 'total_tokens': 87, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'function_call', 'logprobs': None} id='run-6dadedce-b046-4f36-86a2-929f18847et-0'

# forcing model to use function calling
model_with_forced_function = model.bind(functions=[weather_function], function_call={"name":"WeatherSearch"})
print(model_with_forced_function.invoke("what is the weather in sf?"))
# O/P:
# content='' additional_kwargs={'function_call': {'arguments': '{"airport_code":"SFO"}', 'name': 'WeatherSearch'}} response_metadata={'token_usage': {'completion_tokens': 8, 'prompt_tokens': 79, 'total_tokens': 87, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-aeb2c8c5-1df4-40c8-b2f6-182766d5d494-0'

# using in a chain
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}")
])

chain = prompt | model_with_function
print()
print(chain.invoke({"input": "Hi"}))
print(chain.invoke({"input": "what is the weather in sf?"}))
# O/P:
# content='Hello! How can I assist you today?' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 69, 'total_tokens': 80, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-efb2d449-a90c-4dcd-87a7-3d5e5c772ff2-0'
# content='' additional_kwargs={'function_call': {'arguments': '{"airport_code":"SFO"}', 'name': 'WeatherSearch'}} response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 75, 'total_tokens': 93, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'function_call', 'logprobs': None} id='run-29c3b0b2-fecc-40b5-a35b-8c13a32a5990-0'

