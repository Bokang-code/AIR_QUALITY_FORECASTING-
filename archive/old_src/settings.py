import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

if not OPENAQ_API_KEY:
    raise ValueError("OPENAQ_API_KEY not found. Check your .env file.")