from state import AgentState
from typing import Dict
from functions import *
from llms import *
import requests
from bs4 import BeautifulSoup
from langchain_core.messages import SystemMessage

def prediction_agent(state: AgentState) -> Dict:
    """
    Handles queries about price predictions.
    """
    messages = state["messages"]
    last_message = messages[-1]["content"]
    
    # Load predictions if not already loaded
    if state.get("prediction_data") is None:
        state["prediction_data"] = load_predictions()
    
    predictions_df = state["prediction_data"]
    
    # Check if predictions data is empty
    if predictions_df.empty:
        response = llama.invoke([
            HumanMessage(content="""Just Tell the user politely that you couldnt make the predictions,
                          as something went wrong internally and that to try again later and nothing else. Give the message string
                                """)
        ])
        return {"messages": messages + [{"role": "assistant", "content": response.content}]}
    
    # Extract date from user query using our date function
    date_type, date_value = extract_date_from_query(last_message)
    
    # Get the range limits for predictions
    min_date = predictions_df['ds'].min()
    max_date = predictions_df['ds'].max()
    
    if date_type == "range":
        # Using Llama for generating prediction summaries as it has good comprehension
        response = llama.invoke([
            HumanMessage(content=f"""
            The user is asking about cardamom price predictions across a range of dates.
            Here's the first few rows of our predictions:
            
            {predictions_df.head(5).to_string()}
            
            And here's the last few rows of our predictions:
            {predictions_df.tail(5).to_string()}
            
            Generate a helpful summary of the prediction trends over the 90-day period.
            Include the date range, average price, minimum and maximum predicted prices.
            """)
        ])
        return {"messages": messages + [{"role": "assistant", "content": response.content}]}
    
    elif date_type == "none":
        # If no date specified, use the latest prediction
        try:
            # Get the first row (most recent date)
            prediction_row = predictions_df.iloc[0]
            date = prediction_row['ds'].strftime('%Y-%m-%d')
            prediction = float(prediction_row['yhat'])
            lower = float(prediction_row['yhat_lower'])
            upper = float(prediction_row['yhat_upper'])
        except (IndexError, KeyError) as e:
            response = llama.invoke([
                HumanMessage(content=f"""Just Tell the user politely that you couldnt make the predictions, 
                             as something went wrong internally and that to try again later and nothing else.
                              Give the message string""")
            ])
            return {"messages": messages + [{"role": "assistant", "content": response.content}]}
    
    elif date_type in ["specific", "relative"]:
        # For specific or relative dates, check if the date is beyond 90 days
        if date_value:
            try:
                # Convert date_value to timestamp for comparison
                target_date = pd.Timestamp(date_value)
                
                # Check if the target date is beyond our 90-day prediction window
                if target_date > max_date:
                    # Get the latest price from predictions_df (first row)
                    latest_row = predictions_df.iloc[0]
                    latest_date = latest_row['ds'].strftime('%Y-%m-%d')
                    latest_prediction = float(latest_row['yhat'])
                    
                    # Get the furthest prediction (90th day)
                    furthest_row = predictions_df.iloc[-1]
                    furthest_date = furthest_row['ds'].strftime('%Y-%m-%d')
                    furthest_prediction = float(furthest_row['yhat'])
                    
                    response = llama.invoke([
                        HumanMessage(content=f"""
                        The user asked for a prediction on {target_date.strftime('%Y-%m-%d')}, which is beyond our 90-day prediction window.
                        
                        Please explain that we can only provide predictions up to {max_date.strftime('%Y-%m-%d')}.
                        
                        Include the following information:
                        - Latest available price: ₹{latest_prediction:.2f} on {latest_date}
                        - Our furthest prediction: ₹{furthest_prediction:.2f} on {furthest_date}
                        
                        Make the response conversational and helpful, suggesting they ask about dates within our prediction window.
                        """)
                    ])
                    return {"messages": messages + [{"role": "assistant", "content": response.content}]}
                
                # Create a new column with the difference between each date and the target date
                predictions_df['date_diff'] = abs((predictions_df['ds'] - target_date).dt.total_seconds())
                
                # Find the row with the minimum difference
                closest_row = predictions_df.loc[predictions_df['date_diff'].idxmin()]
                
                date = closest_row['ds'].strftime('%Y-%m-%d')
                prediction = float(closest_row['yhat'])
                lower = float(closest_row['yhat_lower'])
                upper = float(closest_row['yhat_upper'])
                
                # Remove the temporary column
                predictions_df.drop('date_diff', axis=1, inplace=True)
                
            except Exception as e:
                response = llama.invoke([
                    HumanMessage(content=f"""
                    I couldn't find a prediction for the date: {date_value.strftime('%Y-%m-%d') if hasattr(date_value, 'strftime') else str(date_value)}
                    
                    Our predictions range from {predictions_df['ds'].min().strftime('%Y-%m-%d')} to {predictions_df['ds'].max().strftime('%Y-%m-%d')}.
                    Please try a date within this range.
                    
                    Technical details: {str(e)}
                    """)
                ])
                return {"messages": messages + [{"role": "assistant", "content": response.content}]}
        else:
            # Fallback if date parsing failed
            response = llama.invoke([
                HumanMessage(content=f"""
                I couldn't identify a specific date in the query: "{last_message}"
                
                Please provide a friendly response asking for clarification on which date they want a prediction for.
                Mention that we have predictions available from {predictions_df['ds'].min().strftime('%Y-%m-%d')} to {predictions_df['ds'].max().strftime('%Y-%m-%d')}.
                """)
            ])
            return {"messages": messages + [{"role": "assistant", "content": response.content}]}
    
    try:
        response = llama.invoke([
            HumanMessage(content=f"""
            Create a friendly, helpful response about the cardamom price prediction for {date}.
            
            Here's the data:
            - Predicted price: ₹{prediction:.2f}
            - Lower bound: ₹{lower:.2f}
            - Upper bound: ₹{upper:.2f}
            
            Make the response conversational and easy to understand for someone in the cardamom industry.
            """)
        ])
    except Exception as e:
        response = llama.invoke([
            HumanMessage(content=f"I encountered an error while generating the prediction response: {str(e)}")
        ])
    
    return {"messages": messages + [{"role": "assistant", "content": response.content}]}


def historical_price_agent(state: AgentState) -> Dict:
    """
    Handles queries about historical cardamom prices.
    """
    messages = state["messages"]
    last_message = messages[-1]["content"]
    
    # Extract date information to focus the search
    date_type, date_value = extract_date_from_query(last_message)
    
    # Formulate a search query based on the date information
    if date_type == "none":
        search_query = "historical cardamom prices trends data"
    elif date_type == "range":
        search_query = f"historical cardamom prices {last_message}"
    else:
        # For specific or relative dates
        if date_value:
            date_str = date_value.strftime('%B %Y')  # Format as "Month Year"
            search_query = f"cardamom prices on {date_str} historical data"
        else:
            search_query = "historical cardamom prices trends data"
    
    # Use DDG search to get historical price information
    search_results = ddg_search.invoke(search_query)
    
    # Process search results with Llama
    response = llama.invoke([
        HumanMessage(content=f"""
        The user asked about historical cardamom prices: "{last_message}"
        
        Here are search results that might contain historical price information:
        {search_results}
        
        Extract and summarize the most relevant historical cardamom price information.
        Include:
        1. Price ranges during the requested period if available
        2. Historical trends and patterns
        3. Factors that affected prices during that period
        4. Any notable price fluctuations or stability
        
        Format this as a helpful historical market report. If you can't find specific 
        historical data for the exact period requested, mention that and provide 
        the closest available historical information.
        """)
    ])
    
    return {"messages": messages + [{"role": "assistant", "content": response.content}]}

def info_agent(state: AgentState) -> Dict:
    """
    Handles queries about cardamom information.
    """
    messages = state["messages"]
    last_message = messages[-1]["content"]
    
    # Use DDG search to get information
    search_query = f"cardamom {last_message}"
    search_results = ddg_search.invoke(search_query)
    
    # Process search results with Llama which is good for factual information
    response = llama.invoke([
        HumanMessage(content=f"""
        The user asked: "{last_message}"
        
        Here are search results about cardamom related to their query:
        {search_results}
        
        Please provide a helpful, accurate response about cardamom based on this information.
        Focus specifically on answering their question with relevant details. 
        If the search results don't provide good information, offer general knowledge about cardamom
        cultivation, varieties, diseases, or processing as relevant to their query.
        """)
    ])
    
    return {"messages": messages + [{"role": "assistant", "content": response.content}]}

#Scrape prices from the spice board website
def spiceboard_prices():
    """
    Extracts historic cardamom prices.
    
    Args:
        date: str : Date in DD-MM-YYYY format or None for latest.
                    Note: This function scrapes the *current* daily price page,
                    which usually shows the latest few days. The date parameter
                    is not used for filtering the scraped data, only the latest
                    available data on the page is returned regardless of the date input.
    """
    url = "https://www.indianspices.com/marketing/price/domestic/daily-price.html"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        
        price_data = []
        
        # Find tables potentially containing cardamom prices
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            if not rows: continue

            # Find the header row by looking for price-related text and common column headers
            header_row = None
            header_indices = {}
            for r in rows:
                headers_text = " ".join([cell.text.strip() for cell in r.find_all(['th', 'td'])]).lower()
                # Look for table headers that seem to indicate price data and relevant columns
                # Check for common header patterns seen in the source HTML
                if ('price' in headers_text and any(col_name in headers_text for col_name in ['date', 'auctioneer', 'market'])) or \
                   ('avg.price (rs./kg)' in headers_text and 'maxprice (rs./kg)' in headers_text): # More specific check for small cardamom table
                    header_row = r
                    headers = [cell.text.strip() for cell in header_row.find_all(['th', 'td'])]
                    # Create a dictionary mapping header text to column index
                    # Handle duplicate headers by taking the first occurrence
                    header_indices = {}
                    for i, header in enumerate(headers):
                         # Normalize header text by stripping whitespace and removing unusual characters if necessary
                         clean_header = header.strip().replace('\n', ' ').replace('\r', '').strip()
                         if clean_header and clean_header not in header_indices:
                             header_indices[clean_header] = i
                    break # Found a potential header row, exit inner loop

            if header_row and header_indices:
                # Determine table type based on specific price headers
                is_small_cardamom_table = 'Avg.Price (Rs./Kg)' in header_indices and 'MaxPrice (Rs./Kg)' in header_indices
                is_large_cardamom_table = 'Price (Rs./Kg)' in header_indices and not is_small_cardamom_table # General Price col usually indicates Large Cardamom if Avg/Max are missing
                
                if not is_small_cardamom_table and not is_large_cardamom_table:
                     # Skip tables that don't match expected cardamom price structures
                     continue

                # Process data rows (skip header and any other non-data rows before it)
                try:
                    data_rows = rows[rows.index(header_row) + 1:]
                except ValueError:
                     continue

                for row in data_rows:
                    cols = row.find_all('td')
                    if not cols: continue # Skip empty rows or rows with only th
                    
                    # Basic check to ensure row looks like data (e.g., starts with a number or a date-like string)
                    if not cols[0].text.strip().isdigit() and not any(d.isdigit() for d in cols[0].text.strip()):
                         continue # Skip rows that don't start with number (Sno) or date

                    entry = {'source': 'Spice Board of India'}
                    
                    try:
                        # Extract common data based on potential header keys
                        date_key = 'Date of Auction' if 'Date of Auction' in header_indices else 'Date'
                        if date_key in header_indices and len(cols) > header_indices[date_key]:
                            entry['date'] = cols[header_indices[date_key]].text.strip()

                        market_key = 'Auctioneer' if 'Auctioneer' in header_indices else 'Market'
                        if market_key in header_indices and len(cols) > header_indices[market_key]:
                            entry['market_or_auctioneer'] = cols[header_indices[market_key]].text.strip()

                        # Extract price(s) and quantities based on table type
                        if is_small_cardamom_table:
                            entry['variety'] = 'Small Cardamom'
                            if 'MaxPrice (Rs./Kg)' in header_indices and len(cols) > header_indices['MaxPrice (Rs./Kg)']:
                                entry['max_price_rs_kg'] = float(cols[header_indices['MaxPrice (Rs./Kg)']].text.strip().replace(',', ''))
                            if 'Avg.Price (Rs./Kg)' in header_indices and len(cols) > header_indices['Avg.Price (Rs./Kg)']:
                                entry['avg_price_rs_kg'] = float(cols[header_indices['Avg.Price (Rs./Kg)']].text.strip().replace(',', ''))
                            if 'Total Qty Arrived (Kgs)' in header_indices and len(cols) > header_indices['Total Qty Arrived (Kgs)']:
                                entry['quantity_arrived_kgs'] = float(cols[header_indices['Total Qty Arrived (Kgs)']].text.strip().replace(',', ''))
                            if 'Qty Sold (Kgs)' in header_indices and len(cols) > header_indices['Qty Sold (Kgs)']:
                                entry['quantity_sold_kgs'] = float(cols[header_indices['Qty Sold (Kgs)']].text.strip().replace(',', ''))

                        elif is_large_cardamom_table:
                            entry['variety'] = 'Large Cardamom'
                            if 'Price (Rs./Kg)' in header_indices and len(cols) > header_indices['Price (Rs./Kg)']:
                                entry['price_rs_kg'] = float(cols[header_indices['Price (Rs./Kg)']].text.strip().replace(',', ''))
                            if 'Type' in header_indices and len(cols) > header_indices['Type']:
                                entry['type'] = cols[header_indices['Type']].text.strip()
                            

                        # Add entry only if it has a date and at least one price field
                        if entry.get('date') and (entry.get('avg_price_rs_kg') is not None or entry.get('price_rs_kg') is not None):
                             price_data.append(entry)
                      

                    except (ValueError, IndexError, KeyError) as e:
                        pass # Skip rows with errors

        if not price_data:
             return "Could not find relevant cardamom price data on the Spice Board website tables."
        
        # Format the output string for the agent, clearly listing the data found
        formatted_output = "Scraped Spice Board Prices (Latest Available):\n"
        for entry in price_data:
             formatted_output += f"- Date: {entry.get('date', 'N/A')}, Variety: {entry.get('variety', 'N/A')}\n"
             formatted_output += f"  Market/Auctioneer: {entry.get('market_or_auctioneer', 'N/A')}\n"
             if 'type' in entry: formatted_output += f"  Type: {entry['type']}\n"
             if 'avg_price_rs_kg' in entry: formatted_output += f"  Avg. Price (Rs./Kg): {entry['avg_price_rs_kg']}\n"
             if 'max_price_rs_kg' in entry: formatted_output += f"  Max Price (Rs./Kg): {entry['max_price_rs_kg']}\n"
             if 'price_rs_kg' in entry: formatted_output += f"  Price (Rs./Kg): {entry['price_rs_kg']}\n"
             if 'quantity_arrived_kgs' in entry: formatted_output += f"  Qty Arrived (Kgs): {entry['quantity_arrived_kgs']}\n"
             if 'quantity_sold_kgs' in entry: formatted_output += f"  Qty Sold (Kgs): {entry['quantity_sold_kgs']}\n"
             formatted_output += "---\n"

        return formatted_output
    except BaseException as e:
        return { 'error' : e }

#fetch price from the general search and spiceboard
def price_fetch_agent(state: AgentState) -> Dict:
    """
    Fetches current cardamom prices from the spice board website.
    """
    messages = state["messages"]
    last_message = messages[-1]["content"]
    
    # Use DDG search to get current prices
    search_results = ddg_search.invoke("current and latest cardamom price today in India")
    spiceboard_prices_results = spiceboard_prices()
    # Using Mistral for price analysis as it might be good at extracting structured data
    response = mistral.invoke([
        HumanMessage(content=f"""
        The user asked about current cardamom prices: "{last_message}"
        
        Here are search results that might contain current price information:
        {search_results}
        Additionally here are the prices from the official spice board website:
        {spiceboard_prices_results}
        
        Extract and summarize the most recent and relevant cardamom price information.
        Include:
        1. Current price ranges in different markets if available
        2. Any recent trends (increasing/decreasing)
        3. Factors affecting current prices
        
        Format this as a helpful market report. If precise current prices aren't available,
        mention that and provide the most recent information available.
        """)
    ])
    
    return {"messages": messages + [{"role": "assistant", "content": response.content}]}

def off_topic_agent(state: AgentState) -> Dict:
    """
    Handles off-topic queries.
    """
    messages = state["messages"]
    
    polite_response = "I'm sorry, I cannot help you with that. I can only answer queries regarding cardamom, including price predictions, cultivation information, diseases, or market prices. Please feel free to ask me anything about cardamom!"
    
    return {"messages": messages + [{"role": "assistant", "content": polite_response}]}


#Greeting s agent to greet the user or onboard
# def greeting_agent(state: AgentState) -> Dict:
#     """
#     Handles greetings and introduces itself as CardamomCare.
#     """
#     messages = state["messages"]
    
#     introduction = """Hello! I'm CardamomCare, your cardamom industry assistant. I can help you with:

# 1. Cardamom price predictions
# 2. Current market prices
# 3. Historical price trends
# 4. Information about cardamom cultivation, diseases, and varieties

# How can I assist you with your cardamom-related questions today?"""
    
#     return {"messages": messages + [{"role": "assistant", "content": introduction}]}
def greeting_agent(state: AgentState) -> Dict:
    """
    Dynamically handles greetings, introductions, and expressions of gratitude.
    """
    messages = state["messages"]
    last_message = messages[-1]["content"] if messages else ""
    
    # Use the LLM to generate an appropriate greeting response
    greeting_response = llama.invoke([
        SystemMessage(content="""You are CardamomCare, a specialized assistant for the cardamom industry.
        Generate a friendly and engaging greeting based on the user's message.
        If it's an initial greeting, introduce yourself and your capabilities briefly.
        If the user is expressing gratitude, respond appropriately.
        Keep your response conversational and warm, but concise.
        Always mention that you can help with: cardamom price predictions, current market prices, 
        historical price trends, and information about cardamom cultivation, diseases, and varieties."""),
        HumanMessage(content=f"User message: {last_message}")
    ])
    
    return {"messages": messages + [{"role": "assistant", "content": greeting_response.content}]}


#Gives the capability of the agent
def capability_agent(state: AgentState) -> Dict:
    """
    Explains what CardamomCare can do.
    """
    messages = state["messages"]
    
    capabilities = """As CardamomCare, I'm specialized in all things cardamom and can help you with:

1. **Price Predictions**: I can forecast cardamom prices for specific dates up to 90 days in the future using advanced modeling.

2. **Current Market Prices**: I can provide you with the latest cardamom market prices from the Spice Board of India and other sources.

3. **Historical Price Analysis**: I can show you historical price trends and patterns to help inform your decisions.

4. **Cardamom Information**: I can provide details about:
   - Different cardamom varieties
   - Cultivation techniques
   - Disease management
   - Harvesting best practices
   - Processing methods
   - Market insights

Whether you're a cardamom farmer, trader, or industry professional, I'm here to provide you with valuable information to support your cardamom-related activities. How can I assist you today?"""
    
    return {"messages": messages + [{"role": "assistant", "content": capabilities}]}