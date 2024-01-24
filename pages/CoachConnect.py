import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import xlrd
st.markdown("# Welcome to CoachConnect 👨‍🏫")
st.markdown("### Empowering Athletes Together\U0001F4AA")
st.sidebar.markdown("# CoachConnect 👨‍🏫")
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
csv_file = r'CoachConnect_files/coach_dashboard_corre.csv'
# csv_file = r"/workspaces/PolarPlotter/CoachConnect_files/coach_dashboard_corre.csv"
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

st.write("### Individual Dashboard :chart_with_upwards_trend:")

left, right = st.columns(2)    
   
with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date, y=CMJ,
                        mode='lines+markers',
                        name='CMJ'))
    fig.update_layout(
        title="CMJ",
        xaxis_title="Date",
        yaxis_title="(N/kg)",
        yaxis_title_font_size = 24, 
        xaxis_title_font_size = 24, 
        hoverlabel_font_size=24,
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
        name='Data'))

    fig.update_layout(
        title="FLEX - EXT",
        xaxis_title="2024",
        yaxis_title="N",
        yaxis_title_font_size = 24, 
        xaxis_title_font_size = 24, 
        hoverlabel_font_size=24,
        font=dict(family="Courier New, monospace", size=26, color="red"),
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


with right:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date, y=CMJ,
                        mode='lines+markers',
                        name='squat iso push'))
    fig.update_layout(
        title="Squat Iso Push",
        xaxis_title="Date",
        yaxis_title="N/kg",
        yaxis_title_font_size = 24, 
        xaxis_title_font_size = 24, 
        hoverlabel_font_size=24,
        title_font=dict(
            family="Courier New, monospace",
            size=36,
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
        yaxis_title_font_size = 24, 
        xaxis_title_font_size = 24, 
        hoverlabel_font_size=24,
        font=dict(family="Courier New, monospace", size=26, color="red"),
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


st.write("### Team Dashboard :chart_with_upwards_trend:")
# team_csv = st.file_uploader("Upload a CSV file", type=["xls", "xlsx"])
team_csv = r"CoachConnect_files/CMJ - SJ - SQUAT ISO HOLD - GOOD MORNING -  EXT.FLEX - ADD.ABD.xls"
if team_csv is not None:
    xls = pd.ExcelFile(team_csv)
    sheet_names = xls.sheet_names
    selected_sheet = st.selectbox("Select a sheet", sheet_names)
    if selected_sheet == "CMJ":
        df = pd.read_excel(xls, skiprows=[0,1,2,3,4,5,6], sheet_name=selected_sheet)
        st.dataframe(df)
    elif selected_sheet == "SJ" or selected_sheet == "SQUAT ISO HOLD":
        df = pd.read_excel(xls, skiprows=[0,1,2,3,4,5, 6], sheet_name=selected_sheet)
        st.dataframe(df)
    else:
        df = pd.read_excel(xls, sheet_name=selected_sheet)
        st.dataframe(df)

    if selected_sheet == "Neck Extension":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Name"], y=df["FORCE/KG"],
                            mode='lines+markers',
                            name='Neck Extension'))
        fig.update_layout(
            title="Neck Extension",
            xaxis_title="Ahtlete",
            yaxis_title="FORCE/KG",
            yaxis_title_font_size = 24, 
            xaxis_title_font_size = 24, 
            hoverlabel_font_size=24,
            title_font=dict(
                family="Courier New, monospace",
                size=42,
                color="white"
                ),
                xaxis=dict(
                tickfont=dict(
                    size=18 
                ) 
                ),
                yaxis=dict(
                tickfont=dict(
                size=18 
            )
        )
        )
        
        st.plotly_chart(fig, use_container_width=True)

if selected_sheet == "Neck flexion":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Name"], y=df["FORCE/KG"],
                        mode='lines+markers',
                        name=selected_sheet))
    fig.update_layout(
        title=selected_sheet,
        xaxis_title="Ahtlete",
        yaxis_title="FORCE/KG",
        yaxis_title_font_size = 24, 
        xaxis_title_font_size = 24, 
        hoverlabel_font_size=24,
        title_font=dict(
            family="Courier New, monospace",
            size=42,
            color="white"
            ),
            xaxis=dict(
            tickfont=dict(
                size=18 
            ) 
            ),
            yaxis=dict(
            tickfont=dict(
            size=18 
        )
    )
    )
    
    st.plotly_chart(fig, use_container_width=True)

if selected_sheet == "CMJ":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Athlete"], y=df["Jump Height (Imp-Mom) [cm]"],
                        mode='lines+markers',
                        name=selected_sheet))
    fig.update_layout(
        title=selected_sheet,
        xaxis_title="Ahtlete",
        yaxis_title="Jump Height (cm)",
        yaxis_title_font_size = 24, 
        xaxis_title_font_size = 24, 
        hoverlabel_font_size=24,
        title_font=dict(
            family="Courier New, monospace",
            size=42,
            color="white"
            ),
            xaxis=dict(
            tickfont=dict(
                size=18 
            ) 
            ),
            yaxis=dict(
            tickfont=dict(
            size=18 
        )
    )
    )    
    st.plotly_chart(fig, use_container_width=True)

if selected_sheet == "SJ":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Athlete"], y=df["Jump Height (Imp-Mom) [cm]"],
                        mode='lines+markers',
                        name=selected_sheet))
    fig.update_layout(
        title=selected_sheet,
        xaxis_title="Ahtlete",
        yaxis_title="Jump Height (cm)",
        yaxis_title_font_size = 24, 
        xaxis_title_font_size = 24, 
        hoverlabel_font_size=24,
        title_font=dict(
            family="Courier New, monospace",
            size=42,
            color="white"
            ),
            xaxis=dict(
            tickfont=dict(
                size=18 
            ) 
            ),
            yaxis=dict(
            tickfont=dict(
            size=18 
        )
    )
    )    
    st.plotly_chart(fig, use_container_width=True)

# ---------- FUNCTIONS ----------
def _reset() -> None:
    st.session_state["title"] = ""
    st.session_state["hovertemplate"] = "%{theta}: %{r}"
    st.session_state["opacity"] = st.session_state["marker_opacity"] = st.session_state[
        "line_smoothing"
    ] = 1
    st.session_state["mode"] = ["lines", "markers"]
    st.session_state["marker_color"] = st.session_state[
        "line_color"
    ] = st.session_state["fillcolor"] = "#636EFA"
    st.session_state["marker_size"] = 6
    st.session_state["marker_symbol"] = "circle"
    st.session_state["line_dash"] = "solid"
    st.session_state["line_shape"] = "linear"
    st.session_state["line_width"] = 2
    st.session_state["fill_opacity"] = 0.5
    # ---------- HEADER ----------
st.title("🕸️ Welcome to Polar Plotter!")
st.subheader("Easily create rich polar/radar/spider plots.")


# ---------- DATA ENTRY ----------
option = st.radio(
    label="Enter data",
    options=(
        "Upload an excel file ⬆️",
        "Add data manually ✍️",
    ),
    help="Uploaded files are deleted from the server when you\n* upload another file\n* clear the file uploader\n* close the browser tab",
)

if option == "Upload an excel file ⬆️":
    if uploaded_file := st.file_uploader(
        label="Upload a file. File should have the format: Label|Value",
        type=["xlsx", "csv", "xls"],
    ):
        input_df = pd.read_excel(uploaded_file)
        st.dataframe(input_df, hide_index=True)

else:
    if option == "Add data manually ✍️":
        df = pd.DataFrame(columns=["Label", "Value"]).reset_index(drop=True)

    input_df = st.data_editor(
        df,
        num_rows="dynamic",
        hide_index=True,
    )
input_df = pd.read_excel(team_csv)


# ---------- SIDEBAR ----------
VERSION = "0.3.1"
with open("sidebar.html", "r", encoding="UTF-8") as sidebar_file:
    sidebar_html = sidebar_file.read().replace("{VERSION}", VERSION)

## ---------- Customization options ----------
with st.sidebar:
    with st.expander("Customization options"):
        title = st.text_input(
            label="Plot title",
            value="Job Requirements" if option == "Play with example data 💡" else "",
            help="Sets the plot title.",
            key="title",
        )

        opacity = st.slider(
            label="Opacity",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Sets the opacity of the trace",
            key="opacity",
        )

        mode = st.multiselect(
            label="Mode",
            options=["lines", "markers"],
            default=["lines", "markers"],
            help='Determines the drawing mode for this scatter trace. If the provided `mode` includes "text" then the `text` elements appear at the coordinates. '
            'Otherwise, the `text` elements appear on hover. If there are less than 20 points and the trace is not stacked then the default is "lines+markers". Otherwise, "lines".',
            key="mode",
        )

        hovertemplate = st.text_input(
            label="Hover template",
            value="%{theta}: %{r}",
            help=r"""Template string used for rendering the information that appear on hover box. Note that this will override `hoverinfo`.
            Variables are inserted using %{variable}, for example "y: %{y}" as well as %{xother}, {%_xother}, {%_xother_}, {%xother_}.
            When showing info for several points, "xother" will be added to those with different x positions from the first point.
            An underscore before or after "(x|y)other" will add a space on that side, only when this field is shown.
            Numbers are formatted using d3-format's syntax %{variable:d3-format}, for example "Price: %{y:$.2f}".
            https://github.com/d3/d3-format/tree/v1.4.5#d3-format for details on the formatting syntax.
            Dates are formatted using d3-time-format's syntax %{variable|d3-time-format}, for example "Day: %{2019-01-01|%A}".
            https://github.com/d3/d3-time-format/tree/v2.2.3#locale_format for details on the date formatting syntax.
            The variables available in `hovertemplate` are the ones emitted as event data described at this link https://plotly.com/javascript/plotlyjs-events/#event-data.
            Additionally, every attributes that can be specified per-point (the ones that are `arrayOk: True`) are available.
            Anything contained in tag `<extra>` is displayed in the secondary box, for example "<extra>{fullData.name}</extra>".
            To hide the secondary box completely, use an empty tag `<extra></extra>`.""",
            key="hovertemplate",
        )

        marker_color = st.color_picker(
            label="Marker color",
            value="#636EFA",
            key="marker_color",
            help="Sets the marker color",
        )

        marker_opacity = st.slider(
            label="Marker opacity",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Sets the marker opacity",
            key="marker_opacity",
        )

        marker_size = st.slider(
            label="Marker size",
            min_value=0,
            max_value=10,
            value=6,
            step=1,
            help="Sets the marker size (in px)",
            key="marker_size",
        )

        marker_symbol = st.selectbox(
            label="Marker symbol",
            index=24,
            options=[
                "arrow",
                "arrow-bar-down",
                "arrow-bar-down-open",
                "arrow-bar-left",
                "arrow-bar-left-open",
                "arrow-bar-right",
                "arrow-bar-right-open",
                "arrow-bar-up",
                "arrow-bar-up-open",
                "arrow-down",
                "arrow-down-open",
                "arrow-left",
                "arrow-left-open",
                "arrow-open",
                "arrow-right",
                "arrow-right-open",
                "arrow-up",
                "arrow-up-open",
                "arrow-wide",
                "arrow-wide-open",
                "asterisk",
                "asterisk-open",
                "bowtie",
                "bowtie-open",
                "circle",
                "circle-cross",
                "circle-cross-open",
                "circle-dot",
                "circle-open",
                "circle-open-dot",
                "circle-x",
                "circle-x-open",
                "cross",
                "cross-dot",
                "cross-open",
                "cross-open-dot",
                "cross-thin",
                "cross-thin-open",
                "diamond",
                "diamond-cross",
                "diamond-cross-open",
                "diamond-dot",
                "diamond-open",
                "diamond-open-dot",
                "diamond-tall",
                "diamond-tall-dot",
                "diamond-tall-open",
                "diamond-tall-open-dot",
                "diamond-wide",
                "diamond-wide-dot",
                "diamond-wide-open",
                "diamond-wide-open-dot",
                "diamond-x",
                "diamond-x-open",
                "hash",
                "hash-dot",
                "hash-open",
                "hash-open-dot",
                "hexagon",
                "hexagon2",
                "hexagon2-dot",
                "hexagon2-open",
                "hexagon2-open-dot",
                "hexagon-dot",
                "hexagon-open",
                "hexagon-open-dot",
                "hexagram",
                "hexagram-dot",
                "hexagram-open",
                "hexagram-open-dot",
                "hourglass",
                "hourglass-open",
                "line-ew",
                "line-ew-open",
                "line-ne",
                "line-ne-open",
                "line-ns",
                "line-ns-open",
                "line-nw",
                "line-nw-open",
                "octagon",
                "octagon-dot",
                "octagon-open",
                "octagon-open-dot",
                "pentagon",
                "pentagon-dot",
                "pentagon-open",
                "pentagon-open-dot",
                "square",
                "square-cross",
                "square-cross-open",
                "square-dot",
                "square-open",
                "square-open-dot",
                "square-x",
                "square-x-open",
                "star",
                "star-diamond",
                "star-diamond-dot",
                "star-diamond-open",
                "star-diamond-open-dot",
                "star-dot",
                "star-open",
                "star-open-dot",
                "star-square",
                "star-square-dot",
                "star-square-open",
                "star-square-open-dot",
                "star-triangle-down",
                "star-triangle-down-dot",
                "star-triangle-down-open",
                "star-triangle-down-open-dot",
                "star-triangle-up",
                "star-triangle-up-dot",
                "star-triangle-up-open",
                "star-triangle-up-open-dot",
                "triangle-down",
                "triangle-down-dot",
                "triangle-down-open",
                "triangle-down-open-dot",
                "triangle-left",
                "triangle-left-dot",
                "triangle-left-open",
                "triangle-left-open-dot",
                "triangle-ne",
                "triangle-ne-dot",
                "triangle-ne-open",
                "triangle-ne-open-dot",
                "triangle-nw",
                "triangle-nw-dot",
                "triangle-nw-open",
                "triangle-nw-open-dot",
                "triangle-right",
                "triangle-right-dot",
                "triangle-right-open",
                "triangle-right-open-dot",
                "triangle-se",
                "triangle-se-dot",
                "triangle-se-open",
                "triangle-se-open-dot",
                "triangle-sw",
                "triangle-sw-dot",
                "triangle-sw-open",
                "triangle-sw-open-dot",
                "triangle-up",
                "triangle-up-dot",
                "triangle-up-open",
                "triangle-up-open-dot",
                "x",
                "x-dot",
                "x-open",
                "x-open-dot",
                "x-thin",
                "x-thin-open",
                "y-down",
                "y-down-open",
                "y-left",
                "y-left-open",
                "y-right",
                "y-right-open",
                "y-up",
                "y-up-open",
            ],
            help="""Sets the marker symbol type. Adding 100 is equivalent to appending "-open" to a symbol name.
            Adding 200 is equivalent to appending "-dot" to a symbol name. Adding 300 is equivalent to appending "-open-dot" or "dot-open" to a symbol name.""",
            key="marker_symbol",
        )

        line_color = st.color_picker(
            label="Line color",
            value="#636EFA",
            key="line_color",
            help="Sets the line color",
        )

        line_dash = st.selectbox(
            label="Line dash",
            options=["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"],
            help="""Sets the dash style of lines.
            Set to a dash type string ("solid", "dot", "dash", "longdash", "dashdot", or "longdashdot") or a dash length list in px (eg "5px,10px,2px,2px").""",
            key="line_dash",
        )

        line_shape = st.selectbox(
            label="Line shape",
            options=["linear", "spline"],
            help="""Determines the line shape. With "spline" the lines are drawn using spline interpolation.
            The other available values correspond to step-wise line shapes.""",
            key="line_shape",
        )

        line_smoothing = st.slider(
            label="Line smoothing",
            min_value=0.0,
            max_value=1.3,
            value=1.0,
            step=0.1,
            help="""Has an effect only if `shape` is set to "spline" Sets the amount of smoothing.
            "0" corresponds to no smoothing (equivalent to a "linear" shape).""",
            key="line_smoothing",
            disabled=line_shape == "linear",
        )

        line_width = st.slider(
            label="Line width",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="""Sets the line width (in px).""",
            key="line_width",
        )

        fillcolor = st.color_picker(
            label="Fill color",
            value="#636EFA",
            key="fillcolor",
            help="Sets the fill color. Defaults to a half-transparent variant of the line color, marker color, or marker line color, whichever is available.",
        )

        fill_opacity = st.slider(
            label="Fill opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="""Sets the fill opacity.""",
            key="fill_opacity",
        )

        rgba = tuple(
            (
                [int(fillcolor.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
                + [fill_opacity]
            )
        )

        st.button(
            "↩️ Reset to defaults",
            on_click=_reset,
            use_container_width=True,
        )

# ---------- VISUALIZATION ----------
import contextlib
with contextlib.suppress(IndexError, NameError):
    labels = list(input_df[input_df.columns[0]])
    # To close the polygon
    # checkbox = st.radio("Select Activity", (input_df.columns[8:]))

    for i in range(8, len(input_df.columns)):
        checkbox = st.checkbox(input_df.columns[i], value=True, key="checkbox" + str(i))
        if checkbox:
            title = input_df.columns[i]
            values = list(input_df[input_df.columns[i]])
            labels = (labels + [labels[0]])[::-1]
            values = (values + [values[0]])[::-1]
            l, r = st.columns(2)
            with l:
                st.write("### " + title)
                data = go.Scatterpolar(
                    r=values,
                    theta=labels,
                    mode="none" if mode == [] else "+".join(mode),
                    opacity=opacity,
                    hovertemplate=hovertemplate + "<extra></extra>",
                    marker_color=marker_color,
                    marker_opacity=marker_opacity,
                    marker_size=marker_size,
                    marker_symbol=marker_symbol,
                    line_color=line_color,
                    line_dash=line_dash,
                    line_shape=line_shape,
                    line_smoothing=line_smoothing,
                    line_width=line_width,
                    fill="toself",
                    fillcolor=f"RGBA{rgba}" if rgba else "RGBA(99, 110, 250, 0.5)",
                )

                layout = go.Layout(
                    title=dict(
                        text=title,
                        x=0.5,
                        xanchor="center",
                    ),
                    paper_bgcolor="rgba(100,100,100,0)",
                    plot_bgcolor="rgba(100,100,100,0)",
                )

                fig = go.Figure(data=data, layout=layout)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    sharing="streamlit",
                    theme="streamlit",

                )
            with r:
                st.write("### " + title)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=labels, y=values,
                                    mode='lines+markers',
                                    name=title))
                fig.update_layout(
                    title=title,               
                    yaxis_title_font_size = 12, 
                    xaxis_title_font_size = 12, 
                    hoverlabel_font_size=20,
                    title_font=dict(
                        family="Courier New, monospace",
                        size=20,
                        color="white"
                        ),
                        xaxis=dict(
                        tickfont=dict(
                            size=12 
                        ) 
                        ),
                        yaxis=dict(
                        tickfont=dict(
                        size=12 
                    )
                )
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
    lcol, rcol = st.columns(2)

    with lcol:
        with st.expander("💾Download static plot"):
            st.write("Download static image from the plot toolbar")
            st.image("save_as_png.png")

    fig.write_html("interactive.html")
    with open("interactive.html", "rb") as file:
        rcol.download_button(
            "💾Download interactive plot",
            data=file,
            mime="text/html",
            use_container_width=True,
        )

