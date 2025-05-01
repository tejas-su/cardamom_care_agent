from typing import Dict,List,Optional
import pandas as pd

class AgentState(Dict):
    """State for the cardamom agent graph."""
    messages: List[Dict]
    next_agent: Optional[str] = None
    prediction_data: Optional[pd.DataFrame] = None
    search_results: Optional[str] = None
    conversation_history: Optional[List[Dict]] = None  # Saving messages in momory memory