import os
from langchain.chat_models import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")

weather_function = convert_pydantic_to_openai_function(WeatherSearch)

class ArtistSearch(BaseModel):
    """Call this to get the names of songs by a particular artist"""
    artist_name: str = Field(description="name of artist to look up")
    n: int = Field(description="number of results")

functions = [
    convert_pydantic_to_openai_function(WeatherSearch),
    convert_pydantic_to_openai_function(ArtistSearch),
]

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=50,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

model_with_functions = model.bind(functions=functions)

print(model_with_functions.invoke("what is the weather in sf?"))
print()
print(model_with_functions.invoke("what are three songs by taylor swift?"))
print()
print(model_with_functions.invoke("hi!"))

# Output:
# content='' additional_kwargs={'function_call': {'arguments': '{"airport_code":"SFO"}', 'name': 'WeatherSearch'}} response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 116, 'total_tokens': 134, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'function_call', 'logprobs': None} id='run-476dd3d9-78fd-4570-bcda-9fbf440e4d6f-0'
# content='' additional_kwargs={'function_call': {'arguments': '{"artist_name":"Taylor Swift","n":3}', 'name': 'ArtistSearch'}} response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 118, 'total_tokens': 140, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'function_call', 'logprobs': None} id='run-401d7bc7-5650-4f25-92b5-93f8ee18a651-0'
# content='Hello! How can I assist you today?' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 111, 'total_tokens': 122, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-eaf3d443-f42a-478e-90b7-70c5d73cb944-0'
