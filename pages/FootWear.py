import streamlit as st
st.markdown("# Welcome to FootWear \U0001F45F")
st.markdown("### FootWear: Smart Strides, Better Rides \U0001F45F")
st.sidebar.markdown("# FootWear \U0001F45F")

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

st.image(persons[1]["image_url"], caption=f"{persons[1]['name']}", use_column_width=True)

expander_whatis = st.expander("What is FootWear?")
with expander_whatis:
    
    st.markdown("""Step into the future of athletic performance with FootWear, which brings computer vision technology to your fingertips, offering personalized insights into the condition of your running shoes.""")
expander_howitworks = st.expander("How does FootWear work?")
with expander_howitworks:
    st.markdown("""
1. SNAP AND UPLOAD: Snap and upload a picture of the sole of your shoe

2. ANALYZE: Let our FootWear computer vision algorithm analyze the wear and tear of your shoe

3. RECEIVE PERSONALIZED FOOTWEAR RECOMMENDATIONS: FootWear doesn't just tell you when it's time to replace your shoes; it's your virtual footwear consultant. The feature intelligently analyzes the wear and tear patterns, providing recommendations on the type of shoes that best suit your unique gait and activity level.

                
But that's not all – FootWear goes beyond the surface. Dive into the details of your biomechanics as the algorithm classifies your pronation style. Whether you overpronate, supinate, or have a normal pronation pattern, FootWear tailors its suggestions to optimize your comfort, performance, and overall foot health.

Say goodbye to guesswork and hello to a personalized shoe experience. Step confidently, step smartly – with FootWear, every stride is a step towards peak performance and comfort.
""")
    
# related research:
expander_research = st.expander("Related Research")
with expander_research:
    st.markdown("""
    Footwear is a critical component of athletic performance and injury prevention. The right pair of shoes can enhance your performance and reduce the risk of injury. However, with the wide variety of athletic shoes available, it can be challenging to find the right pair. FootWear leverages computer vision technology to analyze the wear and tear of your shoes, providing personalized recommendations based on your unique gait and activity level. This feature goes beyond the surface, classifying your pronation style and tailoring its suggestions to optimize your comfort, performance, and overall foot health. With FootWear, every stride is a step towards peak performance and comfort.
    
    Learn more about the research behind FootWear [here](http://cs231n.stanford.edu/reports/2017/pdfs/119.pdf).
                
    """)