import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv

load_dotenv()

llm = ChatNVIDIA(
    base_url=os.getenv("NVIDIA_BASE_URL"),
    api_key=os.getenv("NVIDIA_API_KEY"),
    model="meta/llama-3.1-8b-instruct"
)

print(llm.invoke("Hello world!"))