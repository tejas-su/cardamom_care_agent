from langchain_groq import ChatGroq
from dotenv import load_dotenv 

load_dotenv()
# Initialize LLM options
llama = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.2, max_retries=2)
llamaScout = ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct', temperature=0.2, max_retries=2)
mistral = ChatGroq(model='mistral-saba-24b', temperature=0.2, max_retries=2)