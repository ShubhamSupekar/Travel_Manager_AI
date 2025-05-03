from langchain.document_loaders import WebBaseLoader
import os
from langchain.chat_models import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from typing import Optional
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser


loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
documents = loader.load()
doc = documents[0]
page_content = doc.page_content[:10000]

class Overview(BaseModel):
    """Overview of a section of text."""
    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(description="Provide the language that the content is written in.")
    keywords: str = Field(description="Provide keywords related to the content.")

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    # max_tokens=50,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Note: The model may not return the full output if the max_tokens is too low.
# If you encounter truncated output, consider increasing the max_tokens parameter.
# max_tokens is too low (e.g., max_tokens=50 in your case).
# Model didn't return full output and you will get below error:
# File "C:\Users\OMEN\.conda\envs\DS\Lib\site-packages\langchain_core\output_parsers\openai_functions.py", line 126, in parse_result
# raise OutputParserException(msg) from exc
# langchain_core.exceptions.OutputParserException: Could not parse function call data: Unterminated string starting at: line 1 column 12 (char 11)     
# For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE

overview_tagging_function = [
    convert_pydantic_to_openai_function(Overview)
]

tagging_model = model.bind(
    functions=overview_tagging_function,
    function_call={"name":"Overview"}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])

tagging_chain = prompt | tagging_model | JsonOutputFunctionsParser()

print(tagging_chain.invoke({"input": page_content}))
# O/P:
# {'summary': 'The article discusses the concept of building autonomous agents powered by LLM (large language model) as the core controller. It covers components such as planning, memory, and tool use, along with challenges and references.', 'language': 'English', 'keywords': 'LLM, autonomous agents, planning, memory, tool use, challenges, references'}
