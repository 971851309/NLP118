import os
import json
from crewai import LLM
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
dotenv_path = find_dotenv()
load_dotenv()

# google
file_path = 'gen-lang-client-0184211067-8d635d347db2.json'

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# gemini model
vertex_credentials_json = json.dumps(vertex_credentials)


# define class for LLM

class google_model:

    def gemini_2_flash():
        return LLM(
            model="gemini/gemini-2.0-flash",
            temperature=0.7,
            vertex_credentials=vertex_credentials_json
        )
        
    def gemini_2_flash_lite():
        return LLM(
            model="gemini/gemini-2.0-flash-lite",
            temperature=0.7,
            vertex_credentials=vertex_credentials_json
        )
        
    def gemini_pro():
        return LLM(
            model="gemini/gemini-2.5-pro-exp-03-25",
            temperature=0.7,
            vertex_credentials=vertex_credentials_json
        )
class local_model():
    
    def mistral():
        return LLM(
                model="ollama/mistral:latest",
                base_url="http://localhost:11434",
                temperature=0.7,
            )
        
    def gemma():
        return LLM(
                model="ollama/gemma3:latest",
                base_url="http://localhost:11434",
                temperature=0.7,
            )
        
    def ollama():
        return LLM(
            model="ollama/llama3.2:latest",
            base_url="http://localhost:11434",
            temperature=0.7,
        )
    
    def cogito():
        return LLM(
                model="ollama/cogito:latest",
                base_url="http://localhost:11434",
                temperature=0.7,
            )