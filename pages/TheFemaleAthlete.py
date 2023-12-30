from io import BytesIO
import streamlit as st

st.markdown("# The Female Digital Athlete :woman-running:")
st.sidebar.markdown("# The Female Athlete :woman-lifting-weights:")

persons = [
    {"image_url": "https://drive.google.com/uc?export=download&id=1HG9H2hFB3HSxBQMgP40qIbEs8uLXRK1b", "name": "Coming Soon...", "description": " "},
]  
st.image(persons[0]["image_url"], caption=f"{persons[0]['name']}", width=500)
