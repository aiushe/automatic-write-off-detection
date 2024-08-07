import os
from dotenv import load_dotenv

def load_env_vars():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    return openai_api_key