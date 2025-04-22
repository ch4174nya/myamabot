import os
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
# the above is because langchain_community import needs that variable

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from bs4 import BeautifulSoup
import requests
from config import get_settings
from retrieve_generate import prompt, State, llm
from langgraph.graph import START, StateGraph

logger = get_settings().logger

vector_store = None

def load_data_from_websites(portfolio_site, blog_site):
    response = requests.get(blog_site)
    logger.info(f'Parsing blog page')
    soup = BeautifulSoup(response.content, 'html.parser')

    links = []
    for link in soup.find_all('a', href=True):
        x = link.get('href')
        if '/blog/posts/' in x:
            logger.info(f'Adding link: {x}')
            links.append(f'{portfolio_site}{x}')

    links += [portfolio_site]
    logger.info(f'links: {links}')
    loader = WebBaseLoader(links)
    data = loader.load()
    return data

def load_data_from_resume(path:str):
    # check if path is a URL
    if path.startswith('http') and path.endswith('.pdf'):
        logger.info(f'Getting resume from URL')
        r = requests.get(path, stream=True)
        with open('resume.pdf', 'wb') as f:
            for chunk in r.iter_content(chunk_size=2000):
                f.write(chunk)

            loader = PyPDFLoader('resume.pdf')
            pages = loader.load()
            return pages
    else:
        # read from local file system, only for testing
        logger.info(f'Getting resume from local file')
        loader = PyPDFLoader(path)
        pages = loader.load()
        return pages

def split_documents(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,      # chunk size (characters)
        chunk_overlap = 200,    # chunk overlap
        add_start_index=True,   # track index in original document
    )
    return text_splitter.split_documents(pages)

def embed_and_store_documents(splits):
    # Generate embeddings
    openai_api_key = get_settings().openai_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key

    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    global vector_store
    vector_store = InMemoryVectorStore(embeddings)
    doc_ids = vector_store.add_documents(documents=splits)
    logger.info(f'{len(doc_ids)} documents stored in vector store, such as {doc_ids[3]}: {vector_store.get_by_ids([doc_ids[3]])}')
    return doc_ids, vector_store

def retrieve(state: State) -> State:
    global vector_store
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
    extracted_data = []
    portfolio_site = get_settings().portfolio_site

    # Get personal website URL
    logger.info(f'Portfolio site: {portfolio_site}')

    # Get the blog URL
    blog_site = get_settings().blogs_site
    logger.info(f'Getting blog page: {blog_site}')

    # Get my blogs:
    # TODO: Include personal blog site (Wordpress)

    extracted_data.extend(
        load_data_from_websites(portfolio_site, blog_site)
    )

    # Get my resume
    resume_path = get_settings().resume_url
    logger.info(f'Getting resume from: {resume_path}')
    extracted_data.extend(
        load_data_from_resume(resume_path)
    )

    total_characters = 0
    for page in extracted_data:
        total_characters += len(page.page_content)
    logger.info(f'Extracted Data as {len(extracted_data)} documents; totaling {total_characters} characters')
    print(f'Extracted Data as {len(extracted_data)} documents; totaling {total_characters} characters')

    # Splitting Documents
    all_splits = split_documents(extracted_data)
    logger.info(f'Splitting Documents into {len(all_splits)} documents')
    print(f'Splitting Documents into {len(all_splits)} documents')

    # Embed and Store Documents in a vector store
    document_ids, vector_store = embed_and_store_documents(all_splits)
    logger.info(f'Stored {len(all_splits)} documents in vector store: {vector_store} with IDs: {document_ids[:3]}')
    print(f'Stored {len(all_splits)} documents in vector store: {document_ids[:3]}')

    # RAG:
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    result = graph.invoke({'question': 'What is my name?'})
    print(f'Context: {result["context"]}')
    print(f'Answer: {result["answer"]}')
