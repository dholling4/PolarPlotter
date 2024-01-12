from io import BytesIO
import streamlit as st

st.markdown("# The Female Digital Athlete :woman-running:")
st.sidebar.markdown("# The Female Athlete :woman-lifting-weights:")

persons = [
    {"image_url": "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/the_female_athlete.png", "name": "Coming Soon...", "description": " "},
]  
st.image(persons[0]["image_url"], caption=f"{persons[0]['name']}", width=500)
