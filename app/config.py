from pydantic_settings import BaseSettings
from functools import lru_cache
import logging


class Settings(BaseSettings):
    env_name: str = "local"
    portfolio_site: str = "https://en.wikipedia.org/wiki/Sirius_Black"
    blogs_site: str = "https://harrypotter.fandom.com/wiki/Sirius_Black"
    resume_url: str = "<some path>.pdf"
    # personal_blogs_site: str
    openai_api_key: str = 'sk_xxxxx'

    logging.basicConfig(
        filename="app.log", 
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger: logging.Logger = logging.getLogger(__name__)

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.logger.info(f'Loading settings for: {settings.env_name}')
    return settings