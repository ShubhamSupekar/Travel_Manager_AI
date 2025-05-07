import os
from langchain.tools.render import format_tool_to_openai_function
from langchain_openai import ChatOpenAI
import requests
from pydantic import BaseModel, Field
import datetime
from langchain.agents import tool


# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")


@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°C'


import wikipedia
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


functions = [
    format_tool_to_openai_function(f) for f in [
        search_wikipedia, get_current_temperature
    ]
]


model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    # max_tokens=50,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

model = model.bind(functions=functions)

print(model.invoke("what is the weather in sf right now"))
print(model.invoke("what is langchain"))
# O/P:
# content='' additional_kwargs={'function_call': {'arguments': '{"latitude":37.7749,"longitude":-122.4194}', 'name': 'get_current_temperature'}, 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 26, 'prompt_tokens': 105, 'total_tokens': 131, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-BUQrqbwoChgiFBvZdNhSja3v48yUx', 'service_tier': 'default', 'finish_reason': 'function_call', 'logprobs': None} id='run--378f81eb-feeb-4178-ad01-387a53db5456-0' usage_metadata={'input_tokens': 105, 'output_tokens': 26, 'total_tokens': 131, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
# content='' additional_kwargs={'function_call': {'arguments': '{"query":"Langchain"}', 'name': 'search_wikipedia'}, 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 101, 'total_tokens': 118, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-BUQrr6g94yYOpCRAByoop3DOL9Ipa', 'service_tier': 'default', 'finish_reason': 'function_call', 'logprobs': None} id='run--a545a188-ddcd-403b-bd38-460c91f8ae45-0' usage_metadata={'input_tokens': 101, 'output_tokens': 17, 'total_tokens': 118, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}

from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
])


from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

chain = prompt | model | OpenAIFunctionsAgentOutputParser()
print()
result = chain.invoke({"input": "what is the weather in sf right now"})

print(type(result))
print(result.tool)
print(result.tool_input)
print(get_current_temperature(result.tool_input))

# <class 'langchain_core.agents.AgentActionMessageLog'>
# get_current_temperature
# {'latitude': 37.7749, 'longitude': -122.4194}
# The current temperature is 11.9°C

from langchain.schema.agent import AgentFinish
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "search_wikipedia": search_wikipedia, 
            "get_current_temperature": get_current_temperature,
        }
        return tools[result.tool].run(result.tool_input)
    

chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route

result = chain.invoke({"input": "What is the weather in san francisco right now?"})
print()
print(result)
result = chain.invoke({"input": "What is langchain?"})
print(result)
print(chain.invoke({"input": "hi!"}))
# O/P:
# The current temperature is 11.9°C

# Page: LangChain
# Summary: LangChain is a software framework that helps facilitate the integration of large language models (LLMs) into applications. As a language model integration framework, LangChain's use-cases largely overlap with those of language models in general, including document analysis and summarization, chatbots, and code analysis.  
# Page: Retrieval-augmented generation
# Summary: Retrieval-augmented generation (RAG) is a technique that enables generative artificial intelligence (Gen AI) models to retrieve and incorporate new information. It modifies interactions with a large language model (LLM) so that the model responds to user queries with reference to a specified set of documents, using this information to supplement information from its pre-existing training data. This allows LLMs to use domain-specific and/or updated information. Use cases include providing chatbot access to internal company data or generating responses based on authoritative sources.
# RAG improves large language models (LLMs) by incorporating information retrieval before generating responses. Unlike traditional LLMs that rely on static training data, RAG pulls relevant text from databases, uploaded documents, or web sources. According to Ars Technica, "RAG is a way of improving LLM performance, in essence by blending the LLM process with a web search or other document look-up process to help LLMs stick to the facts." This method helps reduce AI hallucinations, which have led to real-world issues like chatbots inventing policies or lawyers citing nonexistent legal cases.
# By dynamically retrieving information, RAG enables AI to provide more accurate responses without frequent retraining. According to IBM, "RAG also reduces the need for users to continuously train the model on new data and update its parameters as circumstances evolve. In this way, RAG can lower the computational and financial costs of running LLM-powered chatbots in an enterprise setting."
# Beyond efficiency gains, RAG also allows LLMs to include source references in their responses, enabling users to verify information by reviewing cited documents or original sources. This can provide greater transparency, as users can cross-check retrieved content to ensure accuracy and relevance.
# The term "retrieval-augmented generation" (RAG) was first introduced in 2020 by Douwe Kiela, Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, and Sebastian Riedel in their research paper Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, at Meta.
# Page: Model Context Protocol
# Summary: The Model Context Protocol (MCP) is an open standard developed by the artificial intelligence company Anthropic for enabling large language model (LLM) applications to interact with external tools, systems, and data sources. Designed to standardize context exchange between AI assistants and software environments, MCP provides a model-agnostic interface for reading files, executing functions, and handling contextual prompts. It was officially announced and open-sourced by Anthropic in November 2024, with subsequent adoption by major AI providers including OpenAI and Google DeepMind.


# Well, hello there! How can I assist you today?