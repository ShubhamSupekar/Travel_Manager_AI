from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.schema.runnable import RunnableLambda
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

text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)
splits = text_splitter.split_text(doc.page_content)

print("length of split: ",len(splits))

def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list

print(splits[0])

prep = RunnableLambda(
    lambda x: [{"input": doc} for doc in text_splitter.split_text(x)]
)

class Paper(BaseModel):
    """Information about papers mentioned."""
    title: str
    author: Optional[str]


class Info(BaseModel):
    """Information to extract"""
    papers: List[Paper]

paper_extraction_function = [
    convert_pydantic_to_openai_function(Info)
]



model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    # max_tokens=50,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


extraction_model = model.bind(
    functions=paper_extraction_function, 
    function_call={"name":"Info"}
)

template = """A article will be passed to you. Extract from it all papers that are mentioned by this article follow by its author. 

Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.

Do not make up or guess ANY extra information. Only extract what exactly is in the text."""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])

extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")


chain = prep | extraction_chain.map() | flatten

print(chain.invoke(doc.page_content))

# [{'title': 'AutoGPT', 'author': None}, {'title': 'GPT-Engineer', 'author': None}, {'title': 'BabyAGI', 'author': None}, {'title': 'Chain of thought', 'author': 'Wei et al. 2022'}, {'title': 'Tree of Thoughts', 'author': 'Yao et al. 2023'}, {'title': 'LLM+P', 'author': 'Liu et al. 2023'}, {'title': 'ReAct', 'author': 'Yao et al. 2023'}, {'title': 'Reflexion', 'author': 'Shinn & Labash 2023'}, {'title': 'Chain of Hindsight (CoH)', 'author': 'Liu et al. 2023'}, {'title': 'Algorithm Distillation (AD)', 'author': 'Laskin et al. 2023'}, {'title': 'Miller 1956', 'author': None}, {'title': 'Duan et al. 2017', 'author': None}, {'title': 'LSH (Locality-Sensitive Hashing)', 'author': None}, {'title': 'ANNOY (Approximate Nearest Neighbors Oh Yeah)', 'author': None}, {'title': 'HNSW (Hierarchical Navigable Small World)', 'author': None}, {'title': 'FAISS (Facebook AI Similarity Search)', 'author': None}, {'title': 'ScaNN (Scalable Nearest Neighbors)', 'author': None}, {'title': 'MRKL (Karpas et al. 2022)', 'author': 'Karpas et al.'}, {'title': 'TALM (Tool Augmented Language Models; Parisi et al. 2022)', 'author': 'Parisi et al.'}, {'title': 'Toolformer (Schick et al. 2023)', 'author': 'Schick et al.'}, {'title': 'HuggingGPT (Shen et al. 2023)', 'author': 'Shen et al.'}, {'title': 'API-Bank', 'author': 'Li et al. 2023'}, {'title': 'ChemCrow', 'author': 'Bran et al. 2023'}, {'title': 'Boiko et al. (2023)', 'author': 'null'}, {'title': 'Generative Agents Simulation', 'author': 'Park, et al. (2023)'}, {'title': 'Park et al. 2023', 'author': 'null'}, {'title': 'Super Mario: Designing the Perfect Level', 'author': 'John Smith'}, {'title': 'MVC Components in Python', 'author': 'Emily Brown'}, {'title': 'Paper A', 'author': 'Author X'}, {'title': 'Paper B', 'author': 'Author Y'}, {'title': 'Chain of thought prompting elicits reasoning in large language models', 'author': 'Wei et al.'}, {'title': 'Tree of Thoughts: Deliberate Problem Solving with Large Language Models', 'author': 'Yao et al.'}, {'title': 'Chain of Hindsight Aligns Language Models with Feedback', 'author': 'Liu et al.'}, {'title': 'LLM+P: Empowering Large Language Models with Optimal Planning Proficiency', 'author': 'Liu et al.'}, {'title': 'ReAct: Synergizing reasoning and acting in language models', 'author': 'Yao et al.'}, {'title': 'Reflexion: an autonomous agent with dynamic memory and self-reflection', 'author': 'Shinn & Labash'}, {'title': 'In-context Reinforcement Learning with Algorithm Distillation', 'author': 'Laskin et al.'}, {'title': 'MRKL Systems A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning', 'author': 'Karpas et al.'}, {'title': 'Webgpt: Browser-assisted question-answering with human feedback', 'author': 'Nakano et al.'}, {'title': 'Toolformer: Language Models Can Teach Themselves to Use Tools', 'author': 'Schick et al.'}, {'title': 'API-Bank: A Benchmark for Tool-Augmented LLMs', 'author': 'Li et al.'}, {'title': 'HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace', 'author': 'Shen et al.'}, {'title': 'ChemCrow: Augmenting large-language models with chemistry tools', 'author': 'Bran et al.'}, {'title': 'Emergent autonomous scientific research capabilities of large language models', 'author': 'Boiko et al.'}, {'title': 'Generative Agents: Interactive Simulacra of Human Behavior', 'author': 'Joon Sung Park, et al.'}]