from llms import llamaScout
from state import AgentState
from typing import Dict
from langchain_core.messages import HumanMessage


def router(state: AgentState) -> Dict:
    """
    Routes to the appropriate agent based on the query type using AI classification.
    """
    messages = state["messages"]
    last_message = messages[-1]["content"]
    
    # Using LlamaScout for comprehensive routing
    llm_response = llamaScout.invoke([
        HumanMessage(content=f"""
        Analyze the following user query and determine which specialized agent should handle it:
        
        Query: {last_message}
        
        Respond with exactly one of these options:
        - GREETING: If the query is a greeting (hello, hi, etc.), asks who the agent is, or contains thanks/gratitude
        - CAPABILITY: If the query asks what the agent can do or how it can help
        - PREDICTION: If the query is about cardamom price predictions or forecasts
        - INFO: If the query is seeking information about cardamom (cultivation, diseases, etc.)
        - PRICE_FETCH: If the query is about current market prices of cardamom
        - HISTORICAL_PRICE: If the query is about past/historical cardamom prices
        - OFF_TOPIC: If the query is not related to cardamom
        
        Just return the option name, nothing else.
        """)
    ])
    
    agent_type = llm_response.content.strip()
    
    agent_mapping = {
        "GREETING": "greeting_agent",
        "CAPABILITY": "capability_agent",
        "PREDICTION": "prediction_agent",
        "INFO": "info_agent",
        "PRICE_FETCH": "price_fetch_agent",
        "HISTORICAL_PRICE": "historical_price_agent",
        "OFF_TOPIC": "off_topic_agent"
    }
    
    next_agent = agent_mapping.get(agent_type, "off_topic_agent")
    return {"next_agent": next_agent}