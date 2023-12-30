import streamlit as st
import pandas as pd
import numpy as np
import math

st.markdown("# My Activities 🏃‍♂️")
st.sidebar.markdown("# My Activities 🏃‍♂️")

st.write("### Example Charts:")

"""
### Training History
"""
chart_data = pd.DataFrame(
   {
       "Workout": list(range(20)) * 3,
       "time (hrs)": np.abs(np.random.randn(60)),
       "col3": ["Cycling"] * 20 + ["Prehab Exercises"] * 20 + ["Running"] * 20,
   }
)

st.bar_chart(chart_data, x="Workout", y="time (hrs)", color="col3")
left_column, right_column = st.columns(2)
with left_column:
    chosen = st.button("Share with my coach")

with right_column:
    chosen = st.button("Share with my trainer\U0001F510")

chart_data = pd.DataFrame(np.random.normal(8, 2, size=(20, 3)), columns=["Day", "RPE", "Sleep (hrs)"])
chart_data['col4'] = np.random.choice(['Running','Prehab','Cycling'], 20)

st.scatter_chart(
    chart_data,
    x='Day',
    y='RPE',
    color='col4',
    size='Sleep (hrs)',
)
