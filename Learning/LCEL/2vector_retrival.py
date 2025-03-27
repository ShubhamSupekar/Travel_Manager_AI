from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()
print(retriever.get_relevant_documents("where did harrison work?"))
print(retriever.get_relevant_documents("what do bears like to eat"))
