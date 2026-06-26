from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(
    api_key=API_KEY
)