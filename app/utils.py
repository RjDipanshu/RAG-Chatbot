import os
from dotenv import load_dotenv

load_dotenv()

def get_google_api_key():
    return os.getenv("GOOGLE_API_KEY")

def get_port():
    return int(os.getenv("PORT", 8000))
