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
    {"image_url":"https://raw.githubusercontent.com/dholling4/PolarPlotter/main/logo2.png", "name": "The Digital Athlete: Transforming your run using data-driven wearables & AI", "description": " "},
    {"image_url": github_url + "drill-2-skeleton.png", "name": "Motion Analysis", "description": " "},
    {"image_url": github_url+ "coach.jpg", "name": "CoachConnect", "description": " "}, 
    {"image_url": "https://raw.githubusercontent.com/dholling4/PolarPlotter/main//footwear_pics/walk_footwear_CV.png", "name": "Assess worn tread of your running shoe", "description": " "},

]  

col1, col2 = st.columns(2)
with col1:
    st.write("# Welcome to The Digital Athlete!")
    st.write("### Empowering Athletes. :muscle: \n### Elevating Performance. :weight_lifter: \n### Together. :people_holding_hands:")
with col2:
    st.image(persons[0]["image_url"], caption=f"{persons[0]['name']}", width=285)

# =============================================================================
from io import BytesIO
import requests
from PIL import Image

# Function to display MP4 file
def display_video_from_github(repo_url, file_path):
    video_url = f"{repo_url}/raw/main/{file_path}"
    video_response = requests.get(video_url)
    
    if video_response.status_code == 200:
        st.video(video_response.content)
    else:
        st.error(f"Failed to load video from {video_url}")

# GitHub repository URL
github_repo_url = "https://github.com/dholling4/PolarPlotter"

# MP4 file path in the repository
mp4_file_path = "baseline_pics/phone.mp4"

# Display the MP4 file
st.write("#\n")
st.title("Analyze your biomechanics from smartphone video")
display_video_from_github(github_repo_url, mp4_file_path)


# =============================================================================
st.write("#\n")
"""
### What is The Digital Athlete?
"""
expander_whatis = st.expander("Learn More")
with expander_whatis:
    st.write("The Digital Athlete is a platform that *empowers* athletes to take control of their health and performance. Our goal is to provide athletes with the tools they need to perform at their best. We provide a suite of tools that allow athletes to:") 
    st.write("* Track your performance :chart_with_upwards_trend:")
    st.write("* Connect with your coach :runner: :iphone:")
    st.write("* Get footwear recommendations :athletic_shoe:")

st.write("#\n")

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
st.write('#\n')
st.write('#\n')
"""
### Record Activity:
"""
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


# ------- POSE DETECTOR----------------
# import cv2 as cv
# import numpy as np
# import streamlit as st
# import subprocess

# # Clone the GitHub repository
# subprocess.run(["git", "clone", "https://github.com/misbah4064/human-pose-estimation-opencv.git"])

# # Define the body parts and their corresponding indices
# BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
#                "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

# # Define the pairs of body parts that form a pose
# POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

# # Define input dimensions for the network
# width = 368
# height = 368
# inWidth = width
# inHeight = height

# # Load the pre-trained pose detection model
# net = cv.dnn.readNetFromTensorflow("human-pose-estimation-opencv/graph_opt.pb")
# thr = 0.2  # Confidence threshold for the detected keypoints

# # Function to detect poses in a frame
# def poseDetector(frame):
#     frameWidth = frame.shape[1]
#     frameHeight = frame.shape[0]

#     # Prepare the input blob for the network
#     net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
#     out = net.forward()
#     out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

#     assert(len(BODY_PARTS) == out.shape[1])

#     points = []
#     # Iterate over the body parts to extract keypoints
#     for i in range(len(BODY_PARTS)):
#         # Slice heatmap of corresponding body part
#         heatMap = out[0, i, :, :]

#         # Find the maximum confidence and corresponding location
#         _, conf, _, point = cv.minMaxLoc(heatMap)
#         x = (frameWidth * point[0]) / out.shape[3]
#         y = (frameHeight * point[1]) / out.shape[2]
#         points.append((int(x), int(y)) if conf > thr else None)

#     # Connect keypoints to form poses
#     for pair in POSE_PAIRS:
#         partFrom = pair[0]
#         partTo = pair[1]
#         assert(partFrom in BODY_PARTS)
#         assert(partTo in BODY_PARTS)

#         idFrom = BODY_PARTS[partFrom]
#         idTo = BODY_PARTS[partTo]

#         if points[idFrom] and points[idTo]:
#             # Draw line between keypoints
#             cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
#             # Draw keypoints
#             cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
#             cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

#     return frame

# Streamlit App
# st.title("Pose Detection with OpenCV")

# File Uploader
# video_file = st.file_uploader("Upload a video file", type=["mp4"])
# video_file = '/workspaces/PolarPlotter/run_treadmill_outdoors_cut1.mp4'
# if video_file is not None:
#     # Process the uploaded video
#     cap = cv.VideoCapture(video_file)
#     frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv.CAP_PROP_FPS))

#     # Create VideoWriter object to save processed video
#     output_video_path = "/workspaces/PolarPlotter/output_video.mp4"
#     fourcc = cv.VideoWriter_fourcc(*'mp4v')
#     out = cv.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#     # Process each frame in the input video
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Detect poses and overlay skeleton on the frame
#         output_frame = poseDetector(frame)

#         # Write the processed frame to the output video
#         out.write(output_frame)

#     # Release VideoCapture and VideoWriter objects
#     cap.release()
#     out.release()

    # Display the original and processed videos
# st.video("/workspaces/PolarPlotter/output_video.mp4")

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
st.write('#\n')
st.write('#\n')
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
    st.write("#### FootWear \U0001F45F")
    expander_therapist = st.expander("Learn More..")
    with expander_therapist:
        st.write("Snap and upload a picture of the sole of your shoe to get personalized footwear recommendations.")
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
# import pandas as pd

# def save_user_data(user_data):
#     df = pd.DataFrame([user_data])
#     # df.to_csv("Pre-orders/pre_order_data.csv", mode="a", index=False, header=not pd.read_csv("Pre-orders/pre_order_data.csv").exists())
#     df.to_csv("https://raw.githubusercontent.com/dholling4/PolarPlotter/main/Pre-orders/pre_order_data.csv", mode="a", index=False, header=True)
# st.title("Digital Athlete App - Pre-order Signup")

# # Create a form for pre-order signup
# with st.form("pre_order_form"):
#     user_name = st.text_input("Name:")
#     user_email = st.text_input("Email:")
#     user_address = st.text_area("Address:")
#     pre_order_button = st.form_submit_button(label="Pre-order Now")

# if pre_order_button:
#     user_data = {
#         "Name": user_name,
#         "Email": user_email,
#         "Address": user_address
#     }
#     save_user_data(user_data)

#     st.success("Thank you for pre-ordering! We'll notify you when the Digital Athlete app is ready.")

import pandas as pd
import sqlite3

# Function to save user data to SQLite sdatabase
def save_user_data_to_db(user_data):
    # Connect to SQLite database (create a new one if it doesn't exist)
    conn = sqlite3.connect("pre_order_data2.db")
    cursor = conn.cursor()

    # Create a table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pre_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT
        )
    ''')

    # Insert user data into the table
    cursor.execute('''
        INSERT INTO pre_orders (name, email, phone)
        VALUES (?, ?, ?)
    ''', (user_data["Name"], user_data["Email"], user_data["Phone (optional)"]))

    # Commit changes and close the connection
    conn.commit()
    conn.close()

# Streamlit app
st.write('#\n')

st.title("Digital Athlete App - Pre-order Signup")

# Create a form for pre-order signup
with st.form("pre_order_form"):
    user_name = st.text_input("Name:")
    user_email = st.text_input("Email:")
    user_address = st.text_area("Phone (optional):")
    pre_order_button = st.form_submit_button(label="Pre-order Now")

if pre_order_button:
    user_data = {
        "Name": user_name,
        "Email": user_email,
        "Phone (optional)": user_address
    }

    # Save user data to SQLite database
    save_user_data_to_db(user_data)

    st.success("Thank you for pre-ordering! We'll notify you when the Digital Athlete app is ready.")

# ------
# Display existing pre-orders
conn = sqlite3.connect("pre_order_data.db")

existing_pre_orders = pd.read_sql_query("SELECT * FROM pre_orders", conn)
# st.table(existing_pre_orders)
existing_pre_orders.to_sql('pre_orders', conn, if_exists='replace', index=False)

# Close the SQLite connection
conn.close()