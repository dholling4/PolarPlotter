# import streamlit as st
# import pandas as pd
# import numpy as np
# import requests
# from PIL import Image
# from io import BytesIO

# st.sidebar.markdown("# About")

# """
# # The Team
# """
# "##### We bring together a diverse team comprising talented athletes, biomechanical engineers, and computer scientists. Our mission is to harness the power of data-driven biomechanics to benefit athletes."

# # Function to display image with name and description
# # def display_person(image_url, name, description):
# #     st.image(person["image_url"], caption_name = name)
# #     st.write(f"**Name:** {name}")
# #     st.write(f"**Description:** {description}")
    
# # List of person data (replace with your data)
# path = "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/headshots/"
# path2 = "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/"

# persons = [
#     {"image_url": path + "allison.png", "name": "Allison Tanner", "description": "MBA student, Auburn Track & Field"},
#     {"image_url": path + "david_e.png", "name": "David Edmondson", "description": "Master student, Mechanical Engineering, Auburn Track & Field"},
#     {"image_url": path + "david_h.png", "name": "David Hollinger", "description": "PhD candidate, Mechanical Engineering, Nike Sports Research Lab Intern"},
#     {"image_url": path + "avinash.png", "name": "Avinash Baskaran", "description": "PhD student, Mechanical Engineering, Neuromuscular Rehabilitation"},
#     {"image_url": path + "qi_li.png", "name": "Qi Li", "description": "PhD student, Computer Science & Software Engineering"},
#     {"image_url": path + "gulfam.png", "name": "Muhammad Gulfam", "description": "PhD student, Computer Science & Software Engineering"},
# ]  
    
# # Create six columns for each person
# cols = st.columns(6)
# url = 'https://www.linkedin.com/in/david-hollinger1/'
# # st.markdown("[![LinkedIn](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/david-hollinger1/)")
# # st.markdown("[![Title]('https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg')]('https://www.linkedin.com/in/david-hollinger1/')")
# # Display the images as 6 columns
# for i in range(len(persons)):
#     cols[i].image(persons[i]["image_url"], caption=f"{persons[i]['name']}", use_column_width=True)
#     # display empty row except for the 4 and 6th column
#     if i == 0 or i == 1 or i == 2 or i == 4:
#         cols[i].write("")
#     # if i == 2:
#     #     cols[i].markdown("[![LinkedIn](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/david-hollinger1/)")
#     cols[i].write(f"{persons[i]['description']}")

# # Google Drive PNG link
# xray = path2 + "xray.png"

# expander_allison = st.expander("Allison's story")
# with expander_allison:
#     st.write("Ever since I was 12 years old, it was a dream of mine to compete in the SEC for four years.")
#     download_link = f"https://drive.google.com/uc?export=download&id=13rZ4MEJwGjI76QpGgB7TXDPRZ_in9gWJ"
#     video_url = path2 + "Allison%20High%20Jump.MP4"
#     response = requests.get(download_link)
#     st.video(video_url)
#     st.write('I competed for Auburn Track & Field, but my time was cut short. I suffered a torn labrum which required surgery. I came back too quickly, resulting in excessive metatarsal stress in my foot which led to another surgery. Following these two surgeries, I decided to medically retire from competing in the SEC.')
#     persons = [
#     {"image_url": path2 + "xray.png", "name": "X-ray of foot cyst due to excessive metatarsal stress", "description": " "},
#     ]  
#     foot = persons[0]["image_url"]

#     col1, col2 = st.columns(2)
#     col1.image(foot, caption=f"{persons[0]['name']}", use_column_width=True)
#     col2.image(path2 + "torn_labrum.PNG", caption="Surgery from torn labrum", use_column_width=True)
#     # st.image(persons[0]["image_url"], caption=f"{persons[0]['name']}", width=275)
#     st.write(' I am now pursuing an MBA at Auburn University and am excited to be a part of the Digital Athlete team.')

# expander_edmonson = st.expander("David Edmondson's story")
# with expander_edmonson:
#     st.write("I was born in a small town in Jamaica and moved to the United States at the age of 18 to attend Auburn University, competing in the decathlon.")
#     png_url = path2 + "Edmonson_1.png"
#     injury_url = path2 + "edmonson_surgery_pic.PNG"
#     col1, col2 = st.columns(2)
#     col1.image(png_url, caption="Hurdle event during indoor heptathlon", use_column_width=True)
#     col2.image(injury_url, caption="Tight rope surgery to repair torn deltoid in the ankle", use_column_width=True)
#     st.write('During my Senior year, I tore my deltoid ligament in my ankle, requiring tight rope surgery. Following the surgery, I had to relearn how to walk and run. Unfortunately, I never had the chance to compete again in the SEC. I am now a Master student in Mechanical Engineering at Auburn University and am excited to be a part of the Digital Athlete team.')

# expander_hollinger = st.expander("David Hollinger's story")
# with expander_hollinger:
#     jpg_url = path2 + "gmu2.jpg"
#     png_url = path2 + "Marathon.png"
#     col1, col2 = st.columns(2)
#     col1.image(jpg_url, caption="George Mason Cross Country Invitational", use_column_width=True)
#     col2.image(png_url, caption="Bayshore Marathon (Traverse City, MI)", width=250)
#     st.write("I was a walk-on cross country and track runner at George Mason University (Fairfax, VA). Many of my teammates were spectacular, but numerous teammates experienced overuse injuries. Over-use injuries, such as stress fractures, were difficult to predict ahead of time, especially at the elite level. I feared developing an over-use injury. As a result, I under-trained and struggled to find breakthrough. Out of frutstration, I left NCAA competition and joined the Potomac River Running Elite Racing team in DC-VA where I began to run personal bests in the 5k (16:09), 10k (33:25), 10-mile (55:33) and half-marathon (1:15:40). I was able to learn how to properly push myself to new limits. I also ran a 2:50 marathon (Traverse City, MI) in 2018 to qualify for the Boston Marathon. I am now a PhD candidate studying the intersection of biomechanics, wearables, and AI to deliver personalized-insights into training to push athletes to achieve their potential.")

# # Create 2 headshots with advisors
# st.markdown("<h1 style='text-align: center; color: pink;'>Advisors         &       Collaborators</h1>", unsafe_allow_html=True)
# # write collaborator on the right side of the screen

# advisors = [
#     {"image_url": path + "zabala.jpg", "name": "Michael Zabala, PhD", "description": "Auburn Alumni Engineering Council Endowed Associate Professor, Mechanical Engineering, Auburn University"},
#     {"image_url": path + "ford_dyke.jpg", "name": "Ford Dyke, PhD", "description": "Associate Clinical Professor, School of Kinesiology, Auburn University"},
#     {"image_url": path + "iee.png", "name": "", "description": ""},
#     {"image_url": path + "sylvia.jpg", "name": "Sylvia Corre Terente", "description": "Sport Science & Performance Coach Team Arkéa-Samsic"},
# ]
# cols = st.columns(4)
# for i in range(len(advisors)):
#     cols[i].image(advisors[i]["image_url"], caption=f"{advisors[i]['name']}", use_column_width=True)
#     cols[i].write(f"{advisors[i]['description']}")

