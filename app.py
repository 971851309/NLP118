from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent_checkpoint import run_agent  # Make sure the filename matches
import uvicorn

app = FastAPI()

# Allow all origins (for local HTML frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected JSON format
class ReviewRequest(BaseModel):
    name: str
    date: str
    product: str
    review: str

@app.post("/chat")
async def analyze_review(data: ReviewRequest):
    input_data = {
        "cust_name": data.name,
        "purch_date": data.date,
        "product": data.product,
        "review": data.review
    }
    try:
        result = run_agent(input_data)
        return {"reviewed_response": result["reviewed_response"]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
