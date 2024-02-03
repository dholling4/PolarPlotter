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
    {"image_url": path + "worn_out_toe.jpg", "name": "", "description": ""},
]
st.image(persons[0]["image_url"], caption=persons[0]["name"], use_column_width=True)

l, m, r = st.columns(3)
with l:
    st.image(persons[1]["image_url"], caption=persons[0]["name"], use_column_width=True)

with m:
    st.image(persons[2]["image_url"], caption=persons[1]["name"], use_column_width=True)

with r:
    st.image(persons[3]["image_url"], caption=persons[2]["name"], width=130)


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