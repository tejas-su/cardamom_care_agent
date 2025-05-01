from pydantic import BaseModel,Field
from typing import Literal

#Routes of the network
class routes(BaseModel):
    agent : Literal['PREDICTION', 'INFO', 'PRICE_FETCH', 'HISTORICAL_PRICE', 'GREETING', 'CAPABILITY', 'OFF_TOPIC'] = Field(...,description="""
        - PREDICTION: If the query is about cardamom price predictions or forecasts
        - INFO: If the query is seeking information about cardamom (cultivation, diseases, etc.)
        - PRICE_FETCH: If the query is about current market prices of cardamom
        - HISTORICAL_PRICE: If the query is about past/historical cardamom prices
        - GREETING: If the query is a greeting (hello, hi, etc.) or asks who the agent is
        - CAPABILITY: If the query asks what the agent can do or how it can help
        - OFF_TOPIC: If the query is not related to cardamom""")