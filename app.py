import contextlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
# AUTHOR: David Hollinger
import base64
import streamlit as st
import numpy as np
import pandas as pd
import webbrowser
import datetime
import requests
from io import BytesIO
from PIL import Image
import base64

def display_github_image(image_url):
    raw_url = image_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    st.image(raw_url, caption='Image from GitHub', use_column_width=True)

github_url = "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/"

st.sidebar.markdown("# The Digital Athlete ")
download_link = "https://drive.google.com/uc?export=download&id=1MCxkD8d3-JBgi-xVA_IlzMZ_SJ4AOtbE"
david_e = "/workspaces/PolarPlotter/DavidEdmonson_logo_labeled.png"
persons = [
    {"image_url":"https://raw.githubusercontent.com/dholling4/PolarPlotter/main/digital_athlete.png", "name": "The Digital Athlete: Transforming your run using data-driven wearables & AI", "description": " "},
    {"image_url": github_url + "favour_ashe.png", "name": "Motion Analysis", "description": " "},
    {"image_url": github_url+ "coach.jpg", "name": "CoachConnect", "description": " "},
    {"image_url": github_url + "thera_track.jpg", "name": "TheraTrack", "description": " "},

]  

col1, col2 = st.columns(2)
with col1:
    st.write("# Welcome to The Digital Athlete!")
    st.write("### Empowering Athletes. :muscle: \n### Elevating Performance. :weight_lifter: \n### Together. :people_holding_hands:")
with col2:
    st.image(persons[0]["image_url"], caption=f"{persons[0]['name']}", width=285)

"""
### What is The Digital Athlete?
"""
expander_whatis = st.expander("Learn More")
with expander_whatis:
    st.write("The Digital Athlete is a platform that *empowers* athletes to take control of their health and performance. Our goal is to provide athletes with the tools they need to perform at their best. We provide a suite of tools that allow athletes to:") 
    st.write("* Track your performance :chart_with_upwards_trend:")
    st.write("* Connect with your coach :runner: :iphone:")
    st.write("* Communicate with your healthcare providers :male-doctor:")


"""
###
### Key Features
"""

col1, col2, col3 = st.columns(3)
with col1:
    st.write("#### Gait Analysis :runner:")
    expander_gait = st.expander("Learn More")
    with expander_gait:
        st.write("Get metrics on your performance using our AI-powered video analysis tools.") 
        st.image(persons[1]["image_url"], caption=f"{persons[1]['name']}", use_column_width=True)

with col2:
    st.write("#### CoachConnect :male-teacher:")
    expander_coach = st.expander("Learn More..")
    with expander_coach:
        st.write("Connect with your coach to track your performance and get personalized feedback.") 
        st.image(persons[2]["image_url"], caption=f"{persons[2]['name']}", use_column_width=True)

with col3:
    st.write("#### TheraTrack :male-doctor:")
    expander_therapist = st.expander("Learn More..")
    with expander_therapist:
        st.write("Connect with your healthcare provider to track your recovery and get personalized feedback.")
        st.image(persons[3]["image_url"], caption=f"{persons[3]['name']}", use_column_width=True)


st.header("\nFill out your profile")
st.text_input("Name", key="name")
name = st.session_state.name
st.text_input("Age", key="age")
age = st.session_state.age
st.text_input("Height", key="height")
st.text_input("weight (lbs)", key="weight")

"""
##### Select your activity
"""

left_column, right_column = st.columns(2)
df = pd.DataFrame({
    'first column': ['Running', 'Cycling','Golf'],
    })
with left_column:
    chosen_run = st.checkbox("Running")
    chosen_cycle = st.checkbox("Cycling")
    chosen_other = st.checkbox("Other")
    if chosen_run:
        st.write(":runner:")
    if chosen_cycle:
        st.write(":bicyclist:")
    if chosen_other:
        st.write(":weight_lifter:")
        custom_input = st.text_input("Enter your activity")

with right_column:
    chosen = st.radio(
        'Your Level:',
        ("Beginner", "Intermediate", "Elite"))
    
# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)
"""
### Record Activity
"""
# url can have movenet, blazepose or posenet at the end
model = 'movenet'
url_live = 'https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=' + model
url = 'https://storage.googleapis.com/tfjs-models/demos/pose-detection-upload-video/index.html?model=' + model
live_hand_url = 'https://storage.googleapis.com/tfjs-models/demos/hand-pose-detection/index.html?model=mediapipe_hands'
hand_url = 'https://storage.googleapis.com/tfjs-models/demos/hand-pose-detection-upload-video/index.html?model=mediapipe_hands'

left, middle, right = st.columns(3)
with left:
    if st.button('Record video data'):
        webbrowser.open_new_tab(url_live)

with middle:
    if st.button('Record hand video'):
        webbrowser.open_new_tab(live_hand_url)

with right:
    st.button('Record wearable data')

"""
### Upload Activity:
"""
left_column, middle_column, right_column = st.columns(3)

with left_column:
    if st.button('Upload video data'):
        webbrowser.open_new_tab(url)
with middle_column:
    if st.button('Upload hand video'):
        webbrowser.open_new_tab(hand_url)
with right_column:
    st.button('Upload wearable data')
