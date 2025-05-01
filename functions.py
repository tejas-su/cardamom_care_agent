from datetime import datetime,timedelta
from llms import mistral
import pandas as pd
import pickle
from typing import Tuple,Optional
from langchain_core.messages import HumanMessage



# Date handling functions
def parse_relative_date(date_text: str) -> datetime:
    """
    Converts relative date references like 'today', 'tomorrow', 'next week' to actual dates.
    
    Args:
        date_text: A string containing a relative date reference
        
    Returns:
        A datetime object representing the actual date
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Dictionary of common relative date terms and their timedelta values
    relative_dates = {
        "today": timedelta(days=0),
        "tomorrow": timedelta(days=1),
        "day after tomorrow": timedelta(days=2),
        "yesterday": timedelta(days=-1),
        "next week": timedelta(weeks=1),
        "last week": timedelta(weeks=-1),
        "next month": timedelta(days=30),  # Approximation
        "last month": timedelta(days=-30),  # Approximation
        "next year": timedelta(days=365),  # Approximation
        "last year": timedelta(days=-365),  # Approximation
    }
    
    # Handle special cases with specific day names
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    
    date_text = date_text.lower().strip()
    
    # Direct matches from our dictionary
    if date_text in relative_dates:
        return today + relative_dates[date_text]
    
    # Handle "next [weekday]" or "[weekday]"
    for day, day_num in weekdays.items():
        if f"next {day}" in date_text:
            days_ahead = day_num - today.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            return today + timedelta(days=days_ahead)
        
        if date_text == day:
            days_ahead = day_num - today.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            return today + timedelta(days=days_ahead)
    
    # Handle "in X days/weeks/months/years"
    if "in" in date_text:
        parts = date_text.split()
        try:
            if len(parts) >= 3:
                number = int(parts[1])
                unit = parts[2].lower()
                
                if unit.startswith("day"):
                    return today + timedelta(days=number)
                elif unit.startswith("week"):
                    return today + timedelta(weeks=number)
                elif unit.startswith("month"):
                    return today + timedelta(days=number*30)  # Approximation
                elif unit.startswith("year"):
                    return today + timedelta(days=number*365)  # Approximation
        except (ValueError, IndexError):
            pass
    
    # Handle "X days/weeks/months/years from now"
    if "from now" in date_text:
        parts = date_text.split()
        try:
            if len(parts) >= 4:
                number = int(parts[0])
                unit = parts[1].lower()
                
                if unit.startswith("day"):
                    return today + timedelta(days=number)
                elif unit.startswith("week"):
                    return today + timedelta(weeks=number)
                elif unit.startswith("month"):
                    return today + timedelta(days=number*30)  # Approximation
                elif unit.startswith("year"):
                    return today + timedelta(days=number*365)  # Approximation
        except (ValueError, IndexError):
            pass
    
    # Handle "X days/weeks/months/years ago"
    if "ago" in date_text:
        parts = date_text.split()
        try:
            if len(parts) >= 3:
                number = int(parts[0])
                unit = parts[1].lower()
                
                if unit.startswith("day"):
                    return today - timedelta(days=number)
                elif unit.startswith("week"):
                    return today - timedelta(weeks=number)
                elif unit.startswith("month"):
                    return today - timedelta(days=number*30)  # Approximation
                elif unit.startswith("year"):
                    return today - timedelta(days=number*365)  # Approximation
        except (ValueError, IndexError):
            pass
    
    # Default to today if we can't parse the date
    # In a production system, you might want to raise an exception instead
    return today


#Extract s the date from the query and parses it accordingly
def extract_date_from_query(query: str) -> Tuple[str, Optional[datetime]]:
    """
    Extracts date information from a user query.
    
    Args:
        query: The user's query text
        
    Returns:
        A tuple of (date_type, date_value) where:
        - date_type is 'specific', 'relative', 'range', or 'none'
        - date_value is a datetime object or None
    """
    # Using Mistral for date extraction as it might be good at pattern recognition
    date_extraction = mistral.invoke([
        HumanMessage(content=f"""
        Extract the date information from this query: "{query}"
        
        Analyze the query for:
        1. Specific dates (e.g., "April 30, 2025")
        2. Relative dates (e.g., "tomorrow", "next week", "in 3 days")
        3. Date ranges (e.g., "next month", "next 30 days")
        
        Return your answer in this format:
        DATE_TYPE: specific/relative/range/none
        DATE_TEXT: the exact text that refers to the date
        
        Examples:
        For "What will cardamom prices be on May 15th?" return:
        DATE_TYPE: specific
        DATE_TEXT: May 15th
        
        For "How much will cardamom cost tomorrow?" return:
        DATE_TYPE: relative
        DATE_TEXT: tomorrow
        
        For "Predict cardamom prices for the next two weeks" return:
        DATE_TYPE: range
        DATE_TEXT: next two weeks
        
        For "Tell me about cardamom diseases" return:
        DATE_TYPE: none
        DATE_TEXT: 
        """)
    ])
    
    # Parse the LLM response
    response_lines = date_extraction.content.strip().split('\n')
    date_type = "none"
    date_text = ""
    
    for line in response_lines:
        if line.startswith("DATE_TYPE:"):
            date_type = line.split(":", 1)[1].strip().lower()
        elif line.startswith("DATE_TEXT:"):
            date_text = line.split(":", 1)[1].strip()
    
    # Process based on date type
    if date_type == "specific":
        try:
            # Try to parse as a specific date
            date_value = pd.to_datetime(date_text)
            return "specific", date_value
        except:
            # If parsing fails, fall back to today
            return "specific", datetime.now()
    
    elif date_type == "relative":
        # Parse relative date references
        date_value = parse_relative_date(date_text)
        return "relative", date_value
    
    elif date_type == "range":
        # For ranges return the start date
        return "range", None
    
    else:
        # No date mentioned
        return "none", None

# Load the Prophet model predictions from the saved pickle file
def load_predictions():
    """
    Loads the prediction from the saved Prophet modle for 90 days
    """
    with open('prophet_cardamom_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)


    # Step 2: Create future dates for prediction
    future_dates = loaded_model.make_future_dataframe(periods=90)
    forecast = loaded_model.predict(future_dates)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(90)