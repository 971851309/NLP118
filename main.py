from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


# Initialize the language model (
model = OllamaLLM(model="llama3.2")

# Define a prompt template that uses the retrieved examples and the customer query.
template = """
You are an empathetic customer service assistant. Use relevant dialogue examples to help answer question.

Here are some relevant dialogue examples from past conversations: {dialogue_examples}

Customer Query:
{question}

Your empathetic response:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
# Retrieve the top matching documents
    dialogue_examples = retriever.invoke(question)
    
    result = chain.invoke({"dialogue_examples": dialogue_examples, "question": question})

    print("\nResponse:")
    print(result)

 