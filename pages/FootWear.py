import streamlit as st

st.markdown("# Welcome to FootWear \U0001F45F")
st.markdown("### FootWear: Smart Strides, Better Rides \U0001F45F")
st.sidebar.markdown("# FootWear \U0001F45F")

path = "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/footwear_pics/"
persons = [
    # {"image_url": path + "back_footwear.jpg", "name": "coming soon...", "description": ""},
    # {"image_url": path + "run_footwear.jpg", "name": "", "description": ""},
    {"image_url": path + "walk_footwear_CV.png", "name": "", "description": ""},
    {"image_url": path + "wave_rebellion sole_CV.png", "name": "", "description": ""},
    # {"image_url": path + "worn-out-shoes-224x300.jpg", "name": "", "description": ""},
    # {"image_url": path + "worn_out_sole.jpg", "name": "", "description": ""}
    {"image_url": path + "two_worn_shoes.png", "name": "", "description": ""},
    {"image_url": path + "251shoe_report.png", "name": "", "description": ""},
]
st.image(persons[0]["image_url"], caption=persons[0]["name"], use_column_width=True)

# l, m, r = st.columns(3)
# with l:
    # st.image(persons[1]["image_url"], caption=persons[0]["name"], use_column_width=True)

# with m:
    # st.image(persons[2]["image_url"], caption=persons[1]["name"], use_column_width=True)

# with r:
st.image(persons[3]["image_url"], caption=persons[2]["name"], use_column_width=True)


expander_whatis = st.expander("What is FootWear?")
with expander_whatis:
    
    st.markdown("""Step into the future of athletic performance with FootWear, a computer vision algorithm tool to assess worn tread of your running shoe.""")
st.write("## Get SOLEMate Insights!")

youtube_url = "https://www.youtube.com/watch?v=kav_rBFBwtA"
st.video(youtube_url)

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
                and from [Validation of a portable shoe tread scanner to predict slip risk](https://www.sciencedirect.com/science/article/pii/S0022437523000671?via%3Dihub)
                
    """)

expander_technology = st.expander("Technology")
with expander_technology:
    st.markdown("""
    FootWear leverages computer vision technology to analyze the wear and tear of your shoes, providing personalized recommendations based on your unique gait and activity level. This feature goes beyond the surface, classifying your pronation style and tailoring its suggestions to optimize your comfort, performance, and overall foot health. With FootWear, every stride is a step towards peak performance and comfort.
    """)
expander_future = st.expander("Future of FootWear")
with expander_future:
    st.markdown("""
    The future of FootWear is bright. As the technology continues to evolve, FootWear will be able to provide even more personalized recommendations based on a user's unique gait and activity level. With FootWear, every stride is a step towards peak performance and comfort.
    """)
expander_references = st.expander("References")
with expander_references:
    st.markdown("""
    - [Validation of a portable shoe tread scanner to predict slip risk](https://www.sciencedirect.com/science/article/pii/S0022437523000671?via%3Dihub)
    - [Stanford Research](http://cs231n.stanford.edu/reports/2017/pdfs/119.pdf)
    """)
expander_disclaimer = st.expander("Disclaimer")
with expander_disclaimer:
    st.markdown("""
    FootWear is not a medical device and is not intended to diagnose, treat, cure, or prevent any disease. FootWear is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read on FootWear.
    """)
expander_privacy = st.expander("Privacy Policy")

with expander_privacy:
    st.markdown("""
    Your privacy is important to us. To better protect your privacy, we provide this notice explaining our online information practices and the choices you can make about the way your information is collected and used. To make this notice easy to find, we make it available on our homepage and at every point where personally identifiable information may be requested.
    """)
expander_terms = st.expander("Terms of Use")
with expander_terms:
    st.markdown("""
    By using FootWear, you agree to these terms and conditions. Please read them carefully.
    """)
