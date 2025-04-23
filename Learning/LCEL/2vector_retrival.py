import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap


model = ChatOpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.2,
    max_tokens=50,
    request_timeout=10,
    openai_api_key=os.getenv("OPENAI_API_KEY")
) 

output_parser = StrOutputParser()

vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()
# print(retriever.get_relevant_documents("where did harrison work?"))
# print(retriever.get_relevant_documents("what do bears like to eat"))

template =  '''Answer the question based only on the following context:
{context}
Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"],
}) | prompt | model | output_parser

print(chain.invoke({"question":"where did harrison work?"}))