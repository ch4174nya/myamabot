from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import TypedDict, List
from config import get_settings
from langchain.chat_models import init_chat_model

import os

openai_api_key = get_settings().openai_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

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





