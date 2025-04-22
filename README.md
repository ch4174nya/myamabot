# myamabot
An LLM based AMA bot, trained on data from my blog, website and Resume.

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Set up environment variables
Put the following env variables into the .env file at project root.
```
portfolio_site = 'https://en.wikipedia.org/wiki/Sirius_Black'
blogs_site = 'https://harrypotter.fandom.com/wiki/Sirius_Black'
personal_blogs_site = ''
resume_url='<some path>.pdf'
openai_api_key = ''
```

3. Run the application
...