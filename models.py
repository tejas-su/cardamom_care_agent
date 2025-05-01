from pydantic import BaseModel

# Request body schema
class QueryRequest(BaseModel):
    query: str

# Response body schema
class AgentResponse(BaseModel):
    response_text: str