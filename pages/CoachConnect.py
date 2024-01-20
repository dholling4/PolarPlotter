import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
st.markdown("# Welcome to CoachConnect 👨‍🏫")
st.markdown("### Empowering Athletes Together\U0001F4AA")

st.sidebar.markdown("# CoachConnect 👨‍🏫")

#ai coach
st.button("My athletes\U0001F510")

# st.write("#### Coming Soon...")
st.write("#### Individual Athlete Dashboard :chart_with_upwards_trend:")
path2 = "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/"
path = "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/headshots/"
png_url = path + "david_e.png"

l, r = st.columns(2)
with l:
    st.image(png_url, caption="David Edmonson", width=75)
with r:
    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        # st.dataframe(df)
        date = df["Date"].tolist()
        CMJ = df["CMJ (cm)"].tolist()
        squat_iso_push = df["squat iso push (N/kg)"].tolist()
        flex = df["FLEX"].tolist()
        ext = df["EXT"].tolist()
        ratio_flex_ext = df["RATIO (FLEX/EXT)"].tolist()
        abd = df["ABD"].tolist()
        add = df["ADD"].tolist()
        ratio_abd_add = df["RATIO (ABD/ADD)"].tolist()


# selected_columns = st.multiselect('Select columns', df.columns)
# usecols=selected_columns
max_cols=["Max Squat (kg)",	"Max Deadlift (kg)", "Max Hip Thrust (kg)"]
injury_cols = ["Injury", "Date", "Days out",	"RTP"]
competition_cols = ["Competition",	"Date", "Results"]
max_df = df[max_cols].dropna()
injury_df = df[injury_cols].dropna()
competition_df = df[competition_cols].dropna()

st.dataframe(max_df)
col1, col2 = st.columns(2)

with col1:
    st.dataframe(injury_df)

with col2:
    st.dataframe(competition_df)

st.write("#### Team Dashboard :chart_with_upwards_trend:")

left, right = st.columns(2)    
   
with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date, y=CMJ,
                        mode='lines+markers',
                        name='CMJ'))
    fig.update_layout(
        title="CMJ",
        xaxis_title="Date",
        yaxis_title="kg",
        title_font=dict(
            family="Courier New, monospace",
            size=42,
            color="white"
            ),
            xaxis=dict(
            tickfont=dict(
                size=26 
            ) 
            ),
            yaxis=dict(
            tickfont=dict(
            size=26 
        )
    )
    )
            
    st.plotly_chart(fig, use_container_width=True)


    # FLEX AND EXT
    fig = go.Figure()
    # Highlight bars in red if ratio_flex_ext > 1.2
    if ratio_flex_ext[0] > 1.2:
        ratio_color = 'red'
    else:
        ratio_color = 'blue'
    fig.add_trace(go.Bar(
        x=['FLEX', 'EXT', 'RATIO'],
        y=[flex[0], ext[0], ratio_flex_ext[0]],
        text=[np.round(flex[0],1), np.round(ext[0],1), np.round(ratio_flex_ext[0],2)],  # Display the values above each bar
        textposition='auto',  # Automatically position the text above the bars
        marker=dict(color=['lightblue','lightgreen', ratio_color]),
        name='Data',

        title_font=dict(
            family="Courier New, monospace",
            size=42,
            color="white"
            ),
            xaxis=dict(
            tickfont=dict(size=26)),
            yaxis=dict(
            tickfont=dict(size=26)) 
    ))    

    fig.update_layout(
        title="FLEX - EXT",
        xaxis_title="2024",
        yaxis_title="N",
        font=dict(family="Courier New, monospace", size=26, color="RebeccaPurple"),
         xaxis=dict(
        title_font=dict(
            size=28  # Adjust the font size for x-axis label
        )
    ),
    )
    st.plotly_chart(fig, use_container_width=True)


with right:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date, y=squat_iso_push,
                            mode='lines+markers',
                            name='Squat Iso Push'))
    fig.update_layout(
        title="Squat Iso Push",
        xaxis_title="Date",
        yaxis_title="N/kg",
        title_font=dict(
            family="Courier New, monospace",
            size=36,
            color="white" ),

        xaxis=dict(
        tickfont=dict(
            size=26 
        ) 
        ),
        yaxis=dict(
        tickfont=dict(
            size=26 
        )
    )
)
    
    st.plotly_chart(fig, use_container_width=True)

    # ABD AND ADD
    fig = go.Figure()
    # Highlight bars in red if ratio_flex_ext > 1.2
    if ratio_abd_add[0] > 1.2:
        ratio_color = 'red'
    else:
        ratio_color = 'blue'
    fig.add_trace(go.Bar(
        x=['ABD', 'ADD', 'RATIO'],
        y=[abd[0], add[0], ratio_abd_add[0]],
        text=[np.round(abd[0],1), np.round(add[0],1), np.round(ratio_abd_add[0],2)],  # Display the values above each bar
        textposition='auto',  # Automatically position the text above the bars
        marker=dict(color=['lightblue','lightgreen',ratio_color]),
        name='Data'
    ))
    fig.update_layout(
        title="ABD - ADD",
        xaxis_title="2024",
        yaxis_title="N",
        font=dict(family="Courier New, monospace", size=26, color="RebeccaPurple"),
    )
    st.plotly_chart(fig, use_container_width=True)


