# Amazon Chatbot

An AI-powered chatbot that analyzes customer reviews and generates tailored responses using Google Gemini models via Vertex AI. Built with FastAPI, containerized with Docker, and deployed on Render.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Running Locally](#running-locally)  
- [Docker Setup](#docker-setup)  
- [Deployment on Render](#deployment-on-render)  
- [API Reference](#api-reference)  
- [Validation & Testing](#validation--testing)  

---

## Project Structure

```
amazon-chatbot/
├── models/                        # Saved model artifacts
├── .env                           # Environment variables (ignored)
├── Data cleaning+EDA.ipynb        # Data cleaning & initial EDA notebook
├── Data cleaning+EDA.pdf          # PDF export of the notebook
├── EDA.ipynb                      # Additional exploratory analysis
├── agent_checkpoint.py            # AI logic & checkpoint handling
├── app.py                         # FastAPI backend application
├── gen-lang-client-*.js           # Vertex AI credentials (ignored)
├── index.html                     # Frontend HTML interface
├── models.py                      # LLM integration & model management
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules for sensitive files
└── README.md                      # Project documentation
```

- **models/**  
  Directory to store trained LLM wrappers or any serialized model artifacts.  
- **Data cleaning & EDA notebooks**  
  Perform data ingestion, cleaning, exploration, and visualization to understand customer review patterns.  
- **agent_checkpoint.py**  
  Encapsulates the logic for interacting with the LLM (prompt construction, retry strategies, caching).  
- **app.py**  
  FastAPI application that exposes the `/chat` endpoint and serves the frontend.  
- **gen-lang-client-*.js** & **.env**  
  Credentials and API keys for Google Gemini / Vertex AI — these files are excluded via `.gitignore`.  
- **index.html**  
  A single-page interface for end users to submit reviews and view chatbot responses.  
- **models.py**  
  Abstraction layer for loading and managing different LLM clients.  

---

## Prerequisites

- **Python** 3.11+  
- **Docker** (optional, for containerization)  
- **Render** account (for production deployment)  

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/amazon-chatbot.git
   cd amazon-chatbot
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux  
   venv\Scripts\activate         # Windows
   ```

3. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Provision your credentials**  
   – Copy your Vertex AI JSON key into the repo root as `gen-lang-client-<hash>.js`  
   – Create a `.env` file with:
   ```ini
   SERPER_API_KEY=your_serper_key
   GEMINI_API_KEY=your_gemini_key
   ```

---

## Running Locally

Start the FastAPI server with Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open your browser at <http://localhost:8000>.

---

## Docker Setup

Build and run the container:

```bash
# Build
docker build -t amazon-chatbot:latest .

# Run
docker run -d   --name amazon-chatbot   -p 8000:8000   amazon-chatbot:latest
```

**Dockerfile highlights**  
- Base image: `python:3.11-slim`  
- Non-root user for security  
- Installs dependencies from `requirements.txt`  
- Exposes port `8000`  
- Entrypoint via Uvicorn  

---

## Deployment on Render

1. In Render, create a **Docker Web Service** and connect your GitHub repo.  
2. Set the **region** (e.g., Oregon, US).  
3. Add the following **Environment Variables** in Render settings:  
   - `SERPER_API_KEY`  
   - `GEMINI_API_KEY`  
4. Securely upload your `gen-lang-client-*.js` Vertex AI key via Render’s dashboard.  
5. Trigger a deploy and monitor the logs for any errors.

---

## API Reference

### `POST /chat`

- **Description**: Analyze incoming review text and return a tailored, AI-generated response.  
- **Request Body**:  
  ```json
  {
    "review": "I love the quality but shipping was slow."
  }
  ```  
- **Response**:  
  ```json
  {
    "reply": "Thanks for your feedback! We’re glad you love the quality and are working to improve our shipping times..."
  }
  ```

### Static Endpoints

- `GET /` → Serves `index.html`  
- `GET /favicon.ico` → Site icon  

---

## Validation & Testing

- ✅ Verified that `/chat` returns coherent, relevant responses.  
- ✅ Tested data cleaning & EDA workflows in Jupyter notebooks.  
- ✅ Confirmed Docker image builds and runs without errors.  
- ✅ Deployed on Render with no runtime errors and correct environment variable loading.  

---

_Questions or contributions? Feel free to open an issue or submit a pull request!_

