import os
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

print("vector.py has been loaded and executed.")

# Read data
df = pd.read_csv("data/empathetic_chat_cleaned.csv")
# Load embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Specify vectorstore databased
db_location = "./chrome_langchain_db"

add_documents = not os.path.exists(db_location) # If query store it so don't have to re-run

if add_documents:
    documents = []
    ids = []
    
    for idx, row in df.iterrows():
        document = Document(
            page_content= f"Situation: {row['situation']}\nDialogue Text: {row['text']}\nEmotion: {row['emotion_context']}",
            metadata= {
            "conv_id": row["conv_id"],
            "speaker": row["speaker"],
            "role": row["role"],
            "tokens": row["tokens"],
        },
            id=str(idx))

        # Create a Document and assign an id from the row index
        documents.append(document)
        ids.append(str(idx))
        
vector_store = Chroma(
    collection_name="empathetic_dialogues_collection",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"Added {len(documents)} documents to the vector store.")
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)