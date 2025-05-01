import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from build import run_cardamom_agent
from models import *

#initialize the fast api instance
app = FastAPI()


@app.post("/ask", response_model=AgentResponse)
async def ask_cardamom_agent(request: QueryRequest):
    try:
        response_text, _ = run_cardamom_agent(request.query)
        return {"response_text": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#for testing locally
if __name__ == "__main__":
    # Example queries to test
    queries = [
       "wat was the trend last week" ,
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = run_cardamom_agent(query)
        
        # Unpack response
        response_text, message_history = response

        # Prepare JSON structure
        response_json = {
            "response_text": response_text,
            "messages": message_history
        }

        # Print nicely formatted JSON
        print(json.dumps(response_json["response_text"], indent=4))