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



from langchain.tools.render import format_tool_to_openai_function
format_tool_to_openai_function(get_current_temperature)

print(get_current_temperature({"latitude": 19.218330, "longitude": 72.978088}))
print()
# O/P:
# The current temperature is 33.9°C

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

format_tool_to_openai_function(search_wikipedia)

print(search_wikipedia({"query": "langchain"}))
print()
# O/P:
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


