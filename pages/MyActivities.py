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


# # ------------------ #
# # EXAMPLE OF FOOTWEAR RECOMMENDATION
# import streamlit as st
# import plotly.express as px

# def generate_footwear_recommendation_chart(runner_data):
#     # Replace this with your actual data processing and visualization logic
#     # Here, I'm using a dummy scatter plot as an example
#     fig = px.scatter(
#         runner_data,
#         x='stride_length',
#         y='foot_strike_pattern',
#         color='recommended_footwear',
#         size='shoe_size',
#         hover_data=['runner_name'],
#         title='Footwear Recommendation for Runners',
#         labels={'stride_length': 'Stride Length', 'foot_strike_pattern': 'Foot Strike Pattern'},
#     )
#     return fig

# # def main():
# #     st.title('Runner Footwear Recommendation App')
    
# #     # Get runner data (replace this with your actual data source)
# #     runner_data = get_runner_data()

# #     # Display runner data
# #     st.subheader('Runner Data:')
# #     st.dataframe(runner_data)

# #     # Generate and display the footwear recommendation chart
# #     st.subheader('Footwear Recommendation Chart:')
# #     st.plotly_chart(generate_footwear_recommendation_chart(runner_data))

# # if __name__ == "__main__":
# #     main()
