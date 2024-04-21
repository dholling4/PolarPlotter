import streamlit as st
import requests
import time

# Define the URL of your Streamlit app
streamlit_url = "YOUR_STREAMLIT_APP_URL"

# Function to ping the Streamlit app to prevent it from sleeping
def ping_streamlit():
    try:
        response = requests.get(streamlit_url)
        if response.status_code == 200:
            st.success("Streamlit app is awake!")
        else:
            st.error(f"Failed to ping Streamlit app. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error pinging Streamlit app: {e}")

# Ping the Streamlit app every 5 minutes
if __name__ == "__main__":
    while True:
        ping_streamlit()
        time.sleep(3000)  # Ping every 50 minutes