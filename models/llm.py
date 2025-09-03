import os
import getpass
import json
from langchain_google_genai import ChatGoogleGenerativeAI

with open("api_key.json") as f:
    api_info = json.load(f)
    
os.environ["GOOGLE_API_KEY"] = api_info["google"]["api_key"]

def get_llm():
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0,
        max_retries=3,
    )
    
    return llm