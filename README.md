# myamabot
An LLM based AMA bot, trained on data from my blog, website and Resume.

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Set up environment variables
Put the following env variables (substituting for relevant values) into the .env file at project root.
```
portfolio_site = 'https://en.wikipedia.org/wiki/Sirius_Black'
blogs_site = 'https://harrypotter.fandom.com/wiki/Sirius_Black'
personal_blogs_site = ''
resume_url='<some path>.pdf'
openai_api_key = ''
```

3. Run the application
- Ingesting data: `python app/indexing_data.py`
This ingests the data from the sources, storing embeddings in the `InMemoryVectorStore` first. Then the store itself is dumped to file system at the path specified in the `.env` (default: `embeddings_dump.json`).

- Chatting with the bot: `python app/retrieve_generate.py`
This loads the embeddings from the file system and uses them to answer questions (RAG workflow).

#### Features/Todos:
- Streamlit app to chat with the bot
- Deploy 
- Include Wordpress posts into data
