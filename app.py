import contextlib
import plotly.express as px
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
import os

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

st.header("\nFill out your profile")
st.text_input("Name", key="name")
name = st.session_state.name
st.text_input("Age", key="age")
age = st.session_state.age
st.text_input("Height", key="height")
st.text_input("weight (lbs)", key="weight")
chosen_sex = st.radio('Sex:',
             ('Male', 'Female', 'Other'))
if chosen_sex == 'Other':
        custom_input = st.text_input("Enter your preferred sex")



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
# add_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# # )
# """
# ### Record Activity
# """
# # url can have movenet, blazepose or posenet at the end
model = 'movenet'
url_live = 'https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=' + model
url = 'https://storage.googleapis.com/tfjs-models/demos/pose-detection-upload-video/index.html?model=' + model
# live_hand_url = 'https://storage.googleapis.com/tfjs-models/demos/hand-pose-detection/index.html?model=mediapipe_hands'
# hand_url = 'https://storage.googleapis.com/tfjs-models/demos/hand-pose-detection-upload-video/index.html?model=mediapipe_hands'

left, right = st.columns(2)
with left:  
    st.link_button('Record video data', url_live)
with right:
    st.button('Record wearable data')
"""
### Upload Activity:
"""
left_column, right_column = st.columns(2)
with left_column:
    st.link_button('Upload video data', url)
with right_column:
    st.button('Upload wearable data')

# st.write("# Your Run Efficiency Score:")
# dial1, dial2, dial3 = st.columns(3)
# title_font_size = 24
# with dial1:
#   value = 64  # Value to be displayed on the dial (e.g., gas mileage)
#   fig = go.Figure(go.Indicator(
#       mode="gauge+number",
#       value=value,
#       domain={'x': [0, 1], 'y': [0, 1]},
#       gauge=dict(
#           axis=dict(range=[0, 100]),
#           bar=dict(color="white"),
#           borderwidth=2,
#           bordercolor="gray",
#           steps=[
#               dict(range=[0, 25], color="red"),
#               dict(range=[25, 50], color="orange"),
#               dict(range=[50, 75], color="yellow"),
#               dict(range=[75, 100], color="green")
#           ],
#           threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=value)
#       )
#   ))
#   fig.update_layout(
#       title={'text': "Hip Drive", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
#       title_font_size = title_font_size,      
#       font=dict(size=24)
#   )
#   st.plotly_chart(fig, use_container_width=True)
#   st.write("## <div style='text-align: center;'><span style='color: red;'>POOR</span>", unsafe_allow_html=True)
#   # if hip drive is low, recommend hip mobility exercises & strengthening, if really low, also recommend arm swing exercises
#   # recommended drills: SuperMarios, Hill Sprints, single leg hops, deadlifts
#   with st.expander('Hip Drive'):
#       st.write('Hip Drive is the power generated by your hips and glutes to propel you forward during running. Hip drive is important because it helps you run faster and more efficiently. A weak hip drive can lead to overstriding, which can lead to knee pain and shin splints. A strong hip drive can help you run faster and more efficiently.')
#       url = "https://journals.biologists.com/jeb/article/215/11/1944/10883/Muscular-strategy-shift-in-human-running"
#       st.link_button(":book: Read more about the importance of hip drive", url)

# with dial2:
#   value = 75 
#   fig = go.Figure(go.Indicator(
#       mode="gauge+number",
#       value=value,
#       domain={'x': [0, 1], 'y': [0, 1]},
#       gauge=dict(
#           axis=dict(range=[0, 100]),
#           bar=dict(color="white"),
#           borderwidth=2,
#           bordercolor="gray",
#           steps=[
#               dict(range=[0, 25], color="red"),
#               dict(range=[25, 50], color="orange"),
#               dict(range=[50, 75], color="yellow"),
#               dict(range=[75, 100], color="green")
#           ],
#           threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=value)
#       )
#   ))
#   fig.update_layout(
#       title={'text': "Foot Strike Score", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
#       title_font_size = title_font_size,
#       font=dict(size=24)
#   )
#   st.plotly_chart(fig, use_container_width=True)
#   st.write("## <div style='text-align: center;'><span style='color: yellow;'>AVERAGE</span>", unsafe_allow_html=True)

#   # if foot strike is low, recommend drills to increase cadence and reduce overstriding (e.g. high knees, butt kicks, Karaoke, and wind-sprints)
#   with st.expander("Foot Strike Score"):
#       # st.plotly_chart(fig, use_container_width=True)
#       st.write('Foot strike is the first point of contact between your foot and the ground. Foot strike should be on the midfoot, not the heel or the toes. If your foot strike is on your heel, it can lead to overstriding, which can lead to knee pain and shin splints. If your foot strike is on your toes, it can lead to calf pain and achilles tendonitis. A midfoot strike is ideal because it allows your foot to absorb the impact of the ground and propel you forward.')
#       url2 ="https://journals.lww.com/nsca-jscr/abstract/2007/08000/foot_strike_patterns_of_runners_at_the_15_km_point.4"
#       st.link_button(":book: Read more about the importance of foot strike", url2)
# with dial3:
#     value3 = 85  
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=value3,
#         domain={'x': [0, 1], 'y': [0, 1]},
#         gauge=dict(
#             axis=dict(range=[0, 100]),
#             bar=dict(color="white"),
#             borderwidth=2,
#             bordercolor="gray",
#             steps=[
#                 dict(range=[0, 25], color="red"),
#                 dict(range=[25, 50], color="orange"),
#                 dict(range=[50, 75], color="yellow"),
#                 dict(range=[75, 100], color="green")
#             ],
#             threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=value3)
#         )
#     ))
#     fig.update_layout(
#         title={'text': "Arm Swing", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
#         title_font_size = title_font_size,
#         font=dict(size=24)
#     )

#     st.plotly_chart(fig, use_container_width=True)
#     st.write("## <div style='text-align: center;'><span style='color: green;'>GOOD</span>", unsafe_allow_html=True, use_container_width=True)

#     # if arm swing is low, then hip drive is low. Recommend hip mobility exercises and arm swing exercises
#     with st.expander("Arm Swing"):
#         # st.plotly_chart(fig, use_container_width=True)
#         st.write('Arm Swing is important during running because it helps counterbalance the motion of the legs. Arm swing should not cross the midline of the body, but have more of a forward and back rocking motion. Arm swing helps your opposite leg drive forward during toe-off. A strong the arm-swing helps power your hips and knees to drive forward during running. A weak arm swing can lead to a weak hip drive and overstriding.')
#         url = "https://journals.biologists.com/jeb/article/217/14/2456/12120/The-metabolic-cost-of-human-running-is-swinging"
#         st.link_button(":book: Read more about the importance of arm swing", url)



# path = "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/baseline_pics/"
# single_leg_hop_url = path + "single_leg_jump.gif"
# pistol_squat_url = path + "depth_squat.gif"
# st.write("# Recommended Training Program:")
# ex1, ex2 = st.columns(2)
# with ex1:
#     st.image(single_leg_hop_url, caption="Single Leg Hop", use_column_width=True)
# with ex2:
#     st.image(pistol_squat_url, caption="Depth Squat", use_column_width=True)

# # Simulated data for progressive risk trend
# sessions = np.arange(1, 20)  # Running sessions
# biomechanical_risk = np.random.randint(0, 10, size=19)  # Simulated risk scores 

# # Creating a DataFrame for Plotly Express
# data = {'Sessions': sessions, 'Biomechanical Risk': biomechanical_risk}
# # Creating the trend chart
# df = pd.DataFrame(data)
# # Creating the trend chart
# fig = px.line(df, x='Sessions', y='Biomechanical Risk', markers=True,
#                 title='Biomechanical Risk Trend Over Sessions',
#                 labels={'Biomechanical Risk': 'Risk Socre', 'Sessions': 'Running Sessions'},
#                 line_shape='linear')

# # Adding a threshold line for alert
# threshold = 8  # Adjust this threshold based on your criteria
# fig.add_shape(type='line', x0=sessions[0], y0=threshold, x1=sessions[-1], y1=threshold,
#                 line=dict(color='red', dash='dash'), name='Alert Threshold')

# # Adding annotations for alerts
# for session, risk in zip(sessions, biomechanical_risk):
#     if risk > threshold:
#         fig.add_annotation(x=session, y=risk, text='Alert', showarrow=True, arrowhead=2, arrowcolor='red')

# # Display the chart in Streamlit
# # st.write('## Biomechanical Risk Analysis')
# font_size = 28  # Adjust the font size as needed
# fig.update_layout(
#     title_font=dict(size=28),
#     xaxis=dict(title_font=dict(size=font_size), tickfont=dict(size=font_size)),
#     yaxis=dict(title_font=dict(size=font_size), tickfont=dict(size=font_size)),
#     legend=dict(font=dict(size=font_size)),
#     annotations=[dict(font=dict(size=font_size))]
# )
# st.plotly_chart(fig)

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


# TERMS AND CONDITIONS AND PRIVACY POLICY

expander_terms = st.expander("Terms and Conditions")
with expander_terms:

    st.markdown( """
    ## Terms and Conditions
    1. You agree to use this app responsibly.
    2. Any data collected will be used in accordance with our privacy policy.

    """)

    
    st.markdown("""
    ## Privacy Policy
    1. We collect data from you.
    2. We use this data to improve your experience.
    3. Read our full privacy policy [here](https://docs.google.com/document/d/1KQSpmWSaQywkFc8DWdO5kcemPEptHHmSr6XJvWQloSc/edit?usp=sharing).

    """)
    accepted_terms = st.checkbox("I accept the Terms and Conditions")
    if accepted_terms:
        st.success("Thank you for accepting the Terms and Conditions. You can now proceed.")
    else:
        st.warning("Please accept the Terms and Conditions to use the app.")

# PRE-ORDERS!!
import os

def save_user_data(user_data):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "/workspaces/PolarPlotter/pre_orders/existing_pre_orders.csv")

    df = pd.DataFrame([user_data])
    df.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))

st.title("Digital Athlete App - Pre-order Signup")

# Create a form for pre-order signup
with st.form("pre_order_form"):
    user_name = st.text_input("Name:")
    user_email = st.text_input("Email:")
    user_address = st.text_area("Address:")
    pre_order_button = st.form_submit_button(label="Pre-order Now")

if pre_order_button:
    user_data = {
        "Name": user_name,
        "Email": user_email,
        "Address": user_address
    }
    save_user_data(user_data)

    st.success("Thank you for pre-ordering! We'll notify you when the Digital Athlete app is ready.")
