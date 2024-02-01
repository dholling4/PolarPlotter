import streamlit as st
st.markdown("# Welcome to FootWear 👨‍🏫")
st.markdown("### FootWear: Smart Strides, Better Rides.\U0001F45F")
st.sidebar.markdown("# FootWear \U0001F45F")

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