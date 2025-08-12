import streamlit as st
from langchain_core.messages import HumanMessage
from traveller_agent  import create_travel_planner  # your provided code should be saved as travel_planner.py

# Initialize the app and state
if "app" not in st.session_state:
    st.session_state.app = create_travel_planner()
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="centered")
st.title("✈️ AI Travel Planner")

# Sidebar for trip details
with st.sidebar:
    st.header("🧳 Trip Details")
    original_location = st.text_input("Starting Location", "Hyderabad")
    destination_location = st.text_input("Destination", "Goa")
    date_range = st.text_input("Date Range (e.g., 2025-08-19 to 2025-08-25)")
    num_members = st.number_input("Number of Members", min_value=1, step=1, value=1)
    max_price = st.number_input("Max Budget (₹)", min_value=0.0, step=500.0, value=10000.0)
    mode_of_transport = st.selectbox("Mode of Transport", ["car", "bus", "train", "flight"])
    max_results = st.slider("Max Hotel Results", min_value=1, max_value=10, value=5)
    rating = st.slider("Min Hotel Rating", min_value=1.0, max_value=5.0, value=3.0)

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about hotels, weather, or bookings..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Prepare state for LangGraph
    initial_state = {
        'messages': [HumanMessage(content=prompt)],
        'next': '',
        'query': '',
        'current_reasoning': '',
        'destination_location': destination_location,
        'original_location': original_location,
        'date_range': date_range,
        'num_members': num_members,
        'max_price': max_price,
        'mode_of_transport': mode_of_transport,
        'max_results': max_results,
        'rating': rating
    }

    # Invoke the graph
    try:
        result = st.session_state.app.invoke(initial_state)
        if "messages" in result:
            for m in result["messages"]:
                if hasattr(m, "content"):
                    st.session_state.messages.append({"role": "assistant", "content": m.content})
                    st.chat_message("assistant").write(m.content)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.chat_message("assistant").write(error_msg)
