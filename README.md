# Chatbot Model

This project contains a chatbot powered by a machine learning model. Follow the steps below to set up and run it in your local environment.

---

## 🛠️ Setup Instructions

```bash
# 1. Create virtual environment (if not already created)
python -m venv venv

# 2. Activate virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the LLM model
ollama pull llama3.2

# 5. Download the embedding model
ollama pull mxbai-embed-large

# 6. Run the chatbot
python main.py
