from agents import *
from state import AgentState
from langgraph.graph import StateGraph, END
from router import router
from typing import List

# Build the graph
def build_cardamom_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router)
    workflow.add_node("prediction_agent", prediction_agent)
    workflow.add_node("info_agent", info_agent)
    workflow.add_node("price_fetch_agent", price_fetch_agent)
    workflow.add_node("historical_price_agent", historical_price_agent)
    workflow.add_node("greeting_agent", greeting_agent)
    workflow.add_node("capability_agent", capability_agent)  # New node
    workflow.add_node("off_topic_agent", off_topic_agent)
    
    # Add conditional edges from router to appropriate agents
    workflow.add_conditional_edges(
        "router",
        lambda x: x["next_agent"],
        {
            "prediction_agent": "prediction_agent", 
            "info_agent": "info_agent",
            "price_fetch_agent": "price_fetch_agent",
            "historical_price_agent": "historical_price_agent",
            "greeting_agent": "greeting_agent",
            "capability_agent": "capability_agent",  
            "off_topic_agent": "off_topic_agent"
        }
    )
    
    # Connect all agent nodes to END
    workflow.add_edge("prediction_agent", END)
    workflow.add_edge("info_agent", END)
    workflow.add_edge("price_fetch_agent", END)
    workflow.add_edge("historical_price_agent", END)
    workflow.add_edge("greeting_agent", END)
    workflow.add_edge("capability_agent", END)  
    workflow.add_edge("off_topic_agent", END)
    
    # Set the entrypoint
    workflow.set_entry_point("router")
    
    return workflow.compile()

# Create app instance
cardamom_agent = build_cardamom_agent_graph()

#run the cardamom agent
def run_cardamom_agent(query: str, conversation_history: List[Dict] = None):
    """Run the cardamom agent with a query and maintain conversation history."""
    if conversation_history is None:
        conversation_history = []
    
    # Add the new query to the history
    conversation_history.append({"role": "human", "content": query})
    
    # Create state with full conversation history
    state = {
        "messages": [{"role": "human", "content": query}],
        "conversation_history": conversation_history
    }
    
    # Invoke agent
    result = cardamom_agent.invoke(state)
    
    # Add the response to the conversation history
    conversation_history.append({"role": "assistant", "content": result["messages"][-1]["content"]})
    
    # Return the response and updated history
    return result["messages"][-1]["content"], conversation_history