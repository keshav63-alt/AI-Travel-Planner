import os
import re
import requests
from typing import Annotated, Literal, TypedDict
from typing_extensions import TypedDict
# from langchain import LLMChain, config
# config.recursion_limit = 50  # or higher

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize models and tools
groq_model = ChatGroq(model="openai/gpt-oss-120b")
tavily_tool = TavilySearch()

def clean_text(text: str) -> str:
    """Remove <think>...</think> tags from text."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

# State definition
class TravelPlannerState(TypedDict):
    messages: list
    next: str
    query: str
    current_reasoning: str
    destination_location: str
    original_location: str
    date_range: str
    num_members: int
    max_price: float
    mode_of_transport: str
    max_results: int
    rating: float

# Router for supervisor decisions
class RouterDecision(TypedDict):
    next: Literal["information_node", "booking_node", "END"]
    reasoning: str

# Tool definitions
@tool
def search_hotels(
    destination_location: str,
    date_range: str,
    num_members: int,
    max_price: float,
    mode_of_transport: Literal["car", "bus", "train", "flight"],
    max_results: int = 5,
    rating: float = 3.0,
) -> str:
    """
    Search for hotels using the Tavily API.
    
    Args:
        destination_location: The location to search for hotels
        date_range: The date range for the stay (e.g., "2024-08-10 to 2024-08-15")
        num_members: Number of members in the group
        max_price: Maximum price per night
        mode_of_transport: Mode of transport
        max_results: Maximum number of results to return
        rating: Minimum rating of the hotel
    
    Returns:
        str: A string containing the search results
    """
    try:
        # Use Tavily search with hotel-specific query
        search_query = f"hotels in {destination_location} {date_range} {num_members} guests under ${max_price} rating {rating}+"
        results = tavily_tool.invoke({"query": search_query, "max_results": max_results})
        
        if not results:
            return f"No hotels found in {destination_location} within the specified criteria."
        
        return f"Found {len(results)} hotel options in {destination_location}: {results}"
    except Exception as e:
        return f"Error searching for hotels: {str(e)}"

@tool
def get_weather(destination_location: str, date: str) -> str:
    """
    Get the weather forecast for a specific location and date.
    
    Args:
        destination_location: The location to check weather for
        date: The date in YYYY-MM-DD format
    
    Returns:
        str: Weather information
    """
    API_KEY = os.getenv("WEATHER_API_KEY")
    if not API_KEY:
        return "Weather API key not configured."
    
    try:
        # Get coordinates for location
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={destination_location}&limit=1&appid={API_KEY}"
        geo_response = requests.get(geo_url)
        geo_data = geo_response.json()
        
        if not geo_data:
            return f"Location '{destination_location}' not found."

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]

        # Get weather forecast
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        forecast_response = requests.get(forecast_url)
        forecast_data = forecast_response.json()

        # Find matching date in forecast
        for item in forecast_data["list"]:
            if item["dt_txt"].startswith(date):
                temp = item["main"]["temp"]
                desc = item["weather"][0]["description"]
                humidity = item["main"]["humidity"]
                return f"Weather in {destination_location} on {date}: {temp}°C, {desc}, humidity {humidity}%"

        return f"No weather forecast found for {date} in {destination_location}."

    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

@tool
def get_transport_price(
    mode_of_transport: Literal["car", "bus", "train", "flight"],
    num_members: int,
    original_location: str,
    destination_location: str,
    max_price: float,  
) -> str:
    """
    Get the price of transport based on mode, members, and locations.
    
    Args:
        mode_of_transport: The mode of transport
        num_members: Number of members in the group
        original_location: The original location
        destination_location: The destination location
        max_price: Maximum price for the transport
    
    Returns:
        str: Transport pricing information
    """
    try:
        search_query = f"{mode_of_transport} from {original_location} to {destination_location} for {num_members} people under ${max_price}"
        results = tavily_tool.invoke({"query": search_query, "max_results": 3})
        return f"Transport options ({mode_of_transport}): {results}"
    except Exception as e:
        return f"Error getting transport prices: {str(e)}"

@tool
def confirm_hotel_booking(
    hotel_name: str,
    check_in_date: str,
    check_out_date: str,
    num_guests: int,
    total_price: float
) -> str:
    """
    Confirm hotel booking details.
    
    Args:
        hotel_name: Name of the hotel
        check_in_date: Check-in date
        check_out_date: Check-out date
        num_guests: Number of guests
        total_price: Total price for the stay
    
    Returns:
        str: Confirmation message with booking details
    """
    return f"✅ Booking confirmed at {hotel_name} from {check_in_date} to {check_out_date} for {num_guests} guests. Total price: ${total_price:.2f}."

@tool
def cancel_hotel_booking(
    hotel_name: str,
    check_in_date: str,
    check_out_date: str
) -> str:
    """
    Cancel a hotel booking.
    
    Args:
        hotel_name: Name of the hotel
        check_in_date: Check-in date
        check_out_date: Check-out date
    
    Returns:
        str: Confirmation message for cancellation
    """
    return f"❌ Booking at {hotel_name} from {check_in_date} to {check_out_date} has been successfully cancelled."

# Node definitions
def supervisor_node(state: TravelPlannerState) -> Command[Literal['information_node', 'booking_node', '__end__']]:
    """
    Supervisor node that decides which specialized agent should handle the current query.
    """
    system_prompt = """You are a SUPERVISOR agent for a travel planning system.

Available specialized agents:
- INFORMATION_NODE: Searches for hotels, checks weather, gets transport prices
- BOOKING_NODE: Handles hotel booking confirmations and cancellations
- END: Complete the conversation when all requirements are satisfied

Your job is to analyze the conversation and decide which agent should act next.

Rules:
1. If user needs hotel search, weather info, or transport prices → route to "information_node"
2. If user wants to book or cancel hotels → route to "booking_node" 
3. If the user's query is fully answered → route to "END"

Respond with your decision and clear reasoning."""

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add conversation context
    for msg in state.get("messages", []):
        if hasattr(msg, 'content'):
            messages.append({"role": "user", "content": msg.content})
    
    try:
        response = groq_model.with_structured_output(RouterDecision).invoke(messages)
        
        goto = response["next"]
        if goto == "END":
            goto = "__end__"
        
        # Extract query from latest message if available
        query = ""
        if state.get("messages") and len(state["messages"]) >= 1:
            query = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else ""
        
        return Command(
            goto=goto, 
            update={
                'next': goto, 
                'query': query, 
                'current_reasoning': response["reasoning"],
            }
        )
        
    except Exception as e:
        print(f"Supervisor error: {e}")
        return Command(goto="information_node", update={'current_reasoning': f"Error in routing: {e}"})

def information_node(state: TravelPlannerState) -> Command[Literal['supervisor']]:
    """
    Information gathering node for hotels, weather, and transport.
    """
    print("🔍 Information node activated")
    
    system_prompt_text = """You are a travel information specialist. You can:
- Search for hotels with specific criteria
- Check weather forecasts for travel dates
- Get transport pricing estimates

Provide comprehensive, helpful responses based on the user's travel planning needs.
Always use the available tools to get real-time information."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    information_agent = create_react_agent(
        model=groq_model,
        tools=[search_hotels, get_weather, get_transport_price],
        prompt=prompt
    )
    
    try:
        result = information_agent.invoke(state)
        
        # Get the agent's response
        response_content = "I've gathered the requested information."
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            response_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            response_content = clean_text(response_content)

        return Command(
            update={
                "messages": state.get("messages", []) + [
                    AIMessage(content=response_content, name="information_specialist")
                ]
            },
            goto="supervisor",
        )
        
    except Exception as e:
        error_msg = f"Error gathering information: {str(e)}"
        return Command(
            update={
                "messages": state.get("messages", []) + [
                    AIMessage(content=error_msg, name="information_specialist")
                ]
            },
            goto="supervisor",
        )

def booking_node(state: TravelPlannerState) -> Command[Literal['supervisor']]:
    """
    Booking management node for hotel reservations.
    """
    print("📋 Booking node activated")
    
    system_prompt_text = """You are a hotel booking specialist. You can:
- Confirm hotel bookings with all necessary details
- Cancel existing hotel bookings
- Provide booking confirmations and receipts

Always ask for complete information before proceeding with bookings.
Be polite and professional in all interactions."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    booking_agent = create_react_agent(
        model=groq_model,
        tools=[confirm_hotel_booking, cancel_hotel_booking],
        prompt=prompt
    )

    try:
        result = booking_agent.invoke(state)
        
        response_content = "I'm ready to help with your booking needs."
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            response_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            response_content = clean_text(response_content)

        return Command(
            update={
                "messages": state.get("messages", []) + [
                    AIMessage(content=response_content, name="booking_specialist")
                ]
            },
            goto="supervisor",
        )
        
    except Exception as e:
        error_msg = f"Error processing booking: {str(e)}"
        return Command(
            update={
                "messages": state.get("messages", []) + [
                    AIMessage(content=error_msg, name="booking_specialist")
                ]
            },
            goto="supervisor",
        )

# Graph construction
def create_travel_planner():
    """Create and compile the travel planner graph."""
    graph = StateGraph(TravelPlannerState)
    
    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("information_node", information_node)
    graph.add_node("booking_node", booking_node)
    
    # Add edges
    graph.add_edge(START, "supervisor")
    
    return graph.compile()

# Main execution
def main():
    """Main execution function."""
    app = create_travel_planner()
    
    # Test input
    test_message = HumanMessage(
        content="I have to travel from hyderabad to goa from 19th August to 25th August 2025 "
                "for a friend trip. Our budget is 10000 for the whole trip. Can you help me find hotels within budget "
                "and check the weather during these days?"
    )
    
    initial_state = {
        'messages': [test_message],
        'next': '',
        'query': '',
        'current_reasoning': '',
        'destination_location': '',
        'original_location': '',
        'date_range': '',
        'num_members': 1,
        'max_price': 0.0,
        'mode_of_transport': '',
        'max_results': 5,
        'rating': 3.0
    }
    
    try:
        print("🚀 Starting travel planning session...")
        result = app.invoke(initial_state)
        
        print("\n📋 Final Result:")
        for message in result.get('messages', []):
            if hasattr(message, 'content'):
                print(f"💬 {message.content}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running travel planner: {e}")
        return None

if __name__ == "__main__":
    main()