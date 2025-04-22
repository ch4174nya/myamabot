from functools import lru_cache
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from typing_extensions import TypedDict, List
from config import get_settings
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
import os

logger = get_settings().logger

openai_api_key = get_settings().openai_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

def get_vector_store(path: str) -> InMemoryVectorStore:
    # check if the embeddings are available
    if not os.path.exists(path):
        logger.error(f'Vector store not found at {path}')
        logger.error('Please run indexing_data.py first')
        print(f'Vector store not found at {path}. \nPlease run indexing_data.py first')
        exit(1)
    
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    vector_store = InMemoryVectorStore.load(
        path=path, 
        embedding=embeddings
    )
    return vector_store

vector_store = get_vector_store(get_settings().vectorstore_path)

template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Always say "Thanks for asking!" at the end of the answer.
Question: {question} 
Context: {context} 
Helpful Answer:
"""
prompt = PromptTemplate.from_template(template)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State) -> State:
    retrieved_docs = vector_store.similarity_search(state['question'], k=3)
    return {'context': retrieved_docs}

def generate(state: State) -> State:
    docs_content = "\n\n".join(doc.page_content for doc in state['context'])
    messages = prompt.invoke({
        'question': state['question'],
        'context': docs_content
    })
    response = llm.invoke(messages)
    return {'answer': response.content}


if __name__=='__main__':
    # RAG:
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    questions = [
        'What can you tell me about SAFER?', 
        'What is my name?',
        'Does Chaitanya know Python?',
        'Has Chaitanya worked on JavaScript?',
        'List his employers']
    for ques in questions:
        result = graph.invoke({'question': ques})
        logger.info(f'Context: {result["context"]}')
        logger.info(f'Answer: {result["answer"]}')

        print(f'Question: {ques}')
        print(f'Answer: {result["answer"]}')

