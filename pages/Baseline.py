# import contextlib
import pandas as pd
import streamlit as st
# import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import requests

# RUNNING FORM BASELINE EXERCISES TO PERFORM
# HIP --> hip_video, hip_video2, hip_add_video=https://www.youtube.com/watch?v=yCpB5LpS_So ,  https://www.youtube.com/watch?v=FNbmLgpur_c, https://www.youtube.com/watch?v=YQGb-ysmOfU
# Function to display MP4 file

def display_video_from_github(repo_url, file_path):
    video_url = f"{repo_url}/raw/main/{file_path}"
    video_response = requests.get(video_url)
    
    if video_response.status_code == 200:
        st.video(video_response.content)
    else:
        st.error(f"Failed to load video from {video_url}")

# st.video(hip_video)
VERSION = "0.3.1"
path = "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/baseline_pics/"
# THE DIGITAL ATHLETE ========================================
st.markdown("""# Baseline 🧘‍♀️🤸‍♂️📊""")
url = "https://www.acsm.org/docs/default-source/regional-chapter-individual-folders/northland/nacsm--wes-e--fms9a9b0c1f5032400f990d8b57689b0158.pdf?sfvrsn=3668bbe0_0"
st.markdown(" ### Check out the [Functional Movement Screen](%s)" % url)
st.sidebar.markdown("# Baseline 🧘‍♀️🤸‍♂️📊")

squat_url = "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/baseline_pics/depth_squat_enhanced.png"
squat_url = "/workspaces/PolarPlotter/GOOD_SQUAT_SKELETON.gif"
gif_url = path + "balance.gif"

run_front = path + "skeleton_run_frontview2.gif"
run_side = path + "skeleton_run_side_view_enhanced.gif"

gif_url = "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/baseline_pics/balance.gif"
instructions = st.expander("Instructions")
with instructions:
  st.write("1. Record or upload your activity")
  st.write("2. Wait for the video to process")
  st.write("3. View your score and personalized results")

baseline_assessments = st.expander("Baseline Assessments")
cols1, cols2 = st.columns(2)
with baseline_assessments:
  with cols1:
    st.write("#### Gait Analysis")
    github_repo_url = "https://github.com/dholling4/PolarPlotter"
    # MP4 file path in the repository
    mp4_file_path = "baseline_pics/david_treadmill_skeleton.mp4"
    # Display the MP4 file
    st.write("#\n")
    display_video_from_github(github_repo_url, mp4_file_path)
  with cols2:
    st.write("#### Depth Squat")
    github_repo_url = "https://github.com/dholling4/PolarPlotter"
    # MP4 file path in the repository
    mp4_file_path = "baseline_pics/GOOD_SQUAT_SKELETON.mp4"
    # Display the MP4 file
    st.write("#\n")
    display_video_from_github(github_repo_url, mp4_file_path)


    st.write("#### Single Leg Balance")
    st.image(gif_url, caption="Single Leg Balance", width=240)

  st.write("### Baseline assessments include:")
  st.write("* Single Leg Balance")
  st.write("* Depth Squat")
  st.write("* Gait Analysis :runner: :closed_lock_with_key:")

  st.write("### Coming Soon:")
  st.write("* Ankle Mobility")
  st.write("* Hip Mobility")
  st.write("* Core Stability")
  st.write("* Shoulder Mobility")

# # Custom color scheme
color_A = 'rgb(12, 44, 132)'  # Dark blue
color_B = 'rgb(144, 148, 194)'  # Light blue
color_C = 'rgb(171, 99, 250)'  # Purple
color_D = 'rgb(230, 99, 250)'  # Pink
color_E = 'rgb(99, 110, 250)'  # Blue
color_F = 'rgb(25, 211, 243)'  # Turquoise
st.write("# Unlock Your Full Potential: AI-Powered Biomechanics for Personalized Performance!")
st.write("## Get Personalized Results")

col1, col2 = st.columns(2)
with col1:
  chart_data = pd.DataFrame(
    {
        " ": ['Left Knee', 'Right Knee', 'Left Hip', 'Right Hip'],
        "Stability Score": [2.2, 2.6, 3.5, 3.4],
    }
  )
  
  fig = px.bar(chart_data, x=['Left Knee', 'Right Knee', 'Left Hip', 'Right Hip'], y=chart_data["Stability Score"].to_list())
  fig.update_layout(
    xaxis_title = "",
    yaxis_title="Stability Score",
    legend_font_size = 28,
    xaxis_title_font_size = 24, 
    yaxis_title_font_size = 24, 
    hoverlabel_font_size=24,
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),

        xaxis=dict(
        tickfont=dict(size=36 
        ) 
        ),
        yaxis=dict(
        tickfont=dict(
        size=36 )
    ))
  st.plotly_chart(fig, use_container_width=True)

with col2:
  categories = ['Right Knee', 'Right Hip', 'Left Hip',
                  'Left Ankle', 'Right Ankle']
  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
      r=[1.9, 2.3, 3.5, 4.2, 2],
      theta=categories,
      fill='toself',
      line=dict(color=color_A),
      marker=dict(color=color_A, size=10),
      name='Control/Stability'
  ))

  fig.add_trace(go.Scatterpolar(
      r=[3.9, 2.2, 2.4, 1.9, 2],
      theta=categories,
      fill='toself',
      line=dict(color=color_F),
      marker=dict(color=color_F, size=10),
      name='Range of Motion'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 5]
      )
    ),
    # showlegend=True,
    legend=dict(x=0.65, y=0.1),
    font=dict(
      color='white',
      size = 20   # Set font color to white
    ),
  )

  st.plotly_chart(fig, use_container_width=False, width=100)

st.write("# Take your run to the next level!")
# Load data

run1, runner_plots  = st.columns(2)
with run1:
  st.write("### Gait Analysis")
  st.image(run_front, caption="Front View", width=300)
  st.image(run_side, caption="Side View", width=300)

with runner_plots:
  chart_data = pd.DataFrame(
    {"Step Count": list(range(16)), "Left": np.random.randn(16), "Right": np.random.randn(16)}
  )
  st.write('### Foot Strike Score')
  st.bar_chart(
    chart_data, x="Step Count", y=["Left", "Right"], 
    # color=[color_C, "#0000FF"]  # Optional
  )
  # df = load_data(r'C:\Users\dzh0063\OneDrive - Auburn University\Documents\Tiger Cage\Baseline\running_knee_angles_normalized.csv')


  left_knee = [3.5, 6.052, 8.604, 11.156, 12.332, 12.39, 12.448, 12.695, 14.58, 16.465, 18.35, 19.57, 20.44, 21.31, 22.18, 23.05, 23.92, 24.79, 25, 25, 25, 24.559, 23.138, 21.717, 20.296, 19.325, 18.426, 17.527, 16.976, 16.918, 16.86, 16.802, 17.36, 17.94, 18.52, 19.145, 19.812, 20.479, 21.154, 21.937, 22.72, 23.503, 24.376, 25.304, 26.232, 27.025, 27.17, 27.315, 27.46, 29.894, 33.2, 36.506, 39.908, 43.562, 47.216, 50.87, 53.804, 56.588, 59.372, 61.738, 63.42, 65.102, 66.784, 66.927, 66.956, 66.985, 66.3, 64.85, 63.4, 61.91, 59.3, 56.69, 54.08, 52.15, 50.7, 49.25, 47.56, 44.37, 41.18, 37.99, 36.2, 35.04, 33.88, 32.3, 29.4, 26.5, 23.6, 20.93, 18.32, 15.71, 13.5, 12.05, 10.6, 9.15, 8.74, 8.45, 8.16, 8.52, 9.68, 10.84, 12.]
  
  right_knee = [4.9, 6.669, 8.438, 10.207, 11.016, 11.045, 11.074, 11.307, 13.308, 15.309, 17.31, 18.38, 18.96, 19.54, 20.24855071, 21.44987916, 22.65120761, 23.85253606, 23.93515927, 23.66183083, 23.38850238, 22.822, 21.604, 20.386, 19.168, 18.5, 17.92, 17.34, 16.988, 16.959, 16.93, 16.901, 17.488, 18.097, 18.706, 19.45, 20.32, 21.19, 22.04, 22.62, 23.2, 23.78, 24.54, 25.41, 26.28, 27.075, 27.51, 27.945, 28.38, 30.684, 33.7, 36.716, 39.82575785, 43.18163008, 46.5375023, 49.89337452, 52.77471583, 55.55719654, 58.33967724, 60.42732361, 61.37796835, 62.32861308, 63.27925781, 63.20336789, 63.05143837, 62.89950885, 62.30680907, 61.24185546, 60.17690185, 59.08610275, 57.27163015, 55.45715755, 53.64268496, 51.7289717, 49.7452062, 47.76144071, 45.7245872, 43.35593358, 40.98727996, 38.61862634, 36.38269276, 34.2064832, 32.03027365, 29.7188992, 26.98272083, 24.24654245, 21.51036408, 19.64864115, 18.01503703, 16.38143291, 14.87755516, 13.62015748, 12.36275981, 11.10536213, 10.66394325, 10.31667576, 9.96940828, 10.87205426, 13.31305527, 15.75405629, 18.1950573]  
  
  ## Display the chart
  # st.plotly_chart(fig, use_container_width=True)

  chart_data = pd.DataFrame(
    {"Step Count": list(range(len(left_knee))), "Left Knee": left_knee, "Right Knee": right_knee}
  )

# KINEMATICS
st.write('##### Joint Angles')
fs = 30
left_hip_list_x = [33.53389631, 23.60319908, 24.71082991, 31.44837908, 35.71872569, 32.96516151, 28.28836734, 40.9953792, 39.86062259, 33.44414823, 41.47076439, 25.85460365, 40.20734433, 43.9932253, 34.5009277, 29.32044717, 29.53059614, 21.10850017, 13.16118136, 42.41039559, 42.95774915, 41.49441895, 43.0807489, 38.58519925, 30.44002691, 28.2239258, 40.58501604, 41.97357465, 44.90092795, 37.47970917, 35.40727071, 42.80065084, 39.28141527, 30.62938808, 30.86638045, 21.34408458, 14.03436804, 14.80872355, 15.6054373, 20.89026544, 27.79409606, 42.84784067, 34.18172167, 27.93136592, 23.0611644, 34.7116706, 39.66439584, 38.32268776, 40.95874529, 39.52790611, 40.3492191, 43.99573291, 34.52372057, 26.97184025, 19.75252846, 14.5810455, 12.50459463, 17.68137021, 19.12518642, 23.35677677, 26.80732765, 22.53992096, 32.53392676, 29.00108124, 32.64459898, 36.26132052, 42.94534961, 43.66988716, 38.93033199, 44.78470186, 44.11693197, 42.60460079, 30.99932869, 28.07656634, 18.48125998, 11.7676165, 10.50151725, 14.30195246, 19.98869178, 23.24740766, 22.16055184, 31.04804809, 28.28108404, 29.54036391, 37.26289185, 43.88307393, 43.81017342, 42.58629115, 44.58570067, 41.57446955, 34.67608359, 30.48367938, 26.06455147, 15.29782241, 14.43897893, 15.45686709, 16.54996568, 23.87839774, 21.79575576, 34.84248192, 30.02331517, 28.69853475, 30.78808398, 35.39675975, 44.30543326, 42.28800579, 38.62323853, 43.17507745, 43.91494867, 33.62113186, 31.7366487, 29.13153056, 13.79078316, 7.864036973, 18.61875874, 18.12726589, 24.03468475, 28.42438116, 29.78725237, 32.69282359, 26.41422407, 34.29385687, 36.39771945, 42.15621193, 44.21451828, 39.64657897, 44.13566782, 43.32370826, 41.11079839, 32.92536482, 30.74720525, 19.02300658, 13.69007524, 6.812642954, 12.78933252, 15.73457867, 20.35307427, 39.12753915, 30.28039795, 27.01337101, 30.26136371, 32.32609192, 37.26667204, 44.21785092, 43.84530357, 40.36223294, 41.52678493, 40.78510003, 32.66838017, 27.88958285, 21.50134539, 13.99564581, 9.11447025, 44.82410269, 20.1371615, 25.20922098, 40.69251027, 29.75513652, 31.98474818, 29.00873643, 33.22456161, 39.11029684, 41.43555511, 39.47963554, 35.61925248, 42.61342776, 42.68236169, 33.74263833, 32.08062399, 32.18553246, 15.06109906, 11.07342806, 4.995704638, 14.04262182, 44.68124311, 42.45264572, 21.99539729, 32.07216013, 29.69709639, 31.63088262, 34.78498766, 44.3200169, 42.18827131, 42.89356415, 41.5333971, 40.95663346, 36.49648729, 30.93758773, 31.95502352, 36.39595956, 15.78215014, 14.45672484, 20.15271369, 21.58385902, 41.96448377, 32.91886636, 25.7161234, 25.97693594, 32.45653516, 36.76897468, 37.7840693, 44.76747923, 43.64363636, 44.11041891, 41.46010352, 33.43164205, 30.33623782, 32.80668392, 17.38250782, 12.87153597, 12.39614202, 13.39012073, 20.64395774, 42.83184903, 28.28767506, 30.79510535, 30.47389001, 31.54525183, 36.4800443, 40.94045451, 42.46226744, 43.08296369, 44.34570493, 40.71319723, 31.20813476, 28.95069164, 28.75597537, 16.16694179, 15.11659614, 42.79760516, 41.58047058, 21.18260715, 27.7461244, 34.92290118, 30.56732981, 25.52230264, 27.33104502, 35.60753191, 40.37660966, 43.86999536, 42.82397883, 40.75664128, 41.81206516, 33.87176183, 30.86724373, 27.42799601, 16.74048485, 11.23379157, 44.00928312, 7.751710741, 21.6357455, 20.69741986, 21.93927661, 35.38201634, 32.47691507, 31.98132085, 33.73964788, 39.85018041, 41.30632962, 44.34456173, 43.31054297, 41.8082906, 38.35660249, 31.70020241, 31.13296975, 21.99764724, 13.98269421, 11.02302375, 43.12316318, 17.56531887, 24.52042158, 39.03335609, 27.32608782, 28.16847002, 28.18249308, 34.79935298, 37.53402797, 8.426470828, 41.50707735, 44.35685029, 40.15235281, 37.23373901, 29.09625425, 28.0897296, 26.46027802, 16.71007606, 16.68649847, 17.63270282, 20.28153719, 17.49101975, 39.97306823, 33.25426775, 29.25109943, 28.76111863, 31.4884906, 32.83496515, 38.8388699, 44.96944214, 44.16184395, 41.78403062, 41.51636011, 37.4168382, 34.46849445, 28.86623935, 21.98216784, 14.59481254, 6.14812992, 44.76069516, 18.30746555, 39.34292787, 35.58966561, 26.57119627, 27.02718321, 30.12508136, 34.89009772, 35.66234887, 40.10961257, 42.31393165, 40.20681846, 42.2393426, 30.31177974, 27.82691954, 33.69458137, 15.12538954, 4.178903579, 44.92359197, 40.56550992, 25.32026441, 21.89884248, 36.82974855, 31.27478807, 27.30814426, 32.20596541, 34.39741846, 37.95954392, 44.99068749, 42.87867902, 42.66811125, 34.57827368, 34.493955, 32.72163913, 33.64816712, 16.41584451, 11.37893962, 43.77401781, 12.1107742, 15.51938538, 23.42276476, 22.2262684, 26.54283276, 27.17153831, 33.31046306, 35.21733044, 34.21020358, 44.75615839, 42.7055645, 41.28503993, 38.04523472, 38.24964355, 30.28766225, 28.5044385, 31.19175074, 14.7182323, 16.72894939, 16.44930339, 21.6849663, 23.10101859, 40.1433722, 28.20050926, 30.57031842, 31.65182871, 32.04789844, 41.30662656, 42.09292162, 43.50567943, 40.0949953, 40.83884384, 33.30515478, 29.70954083, 27.6663512, 21.45267022, 14.63905549, 10.3853176, 16.06787457, 43.66462563, 20.84927058, 22.09729842, 29.11657631, 29.31671525, 30.68366658, 30.13373694, 37.034661, 41.06764124, 43.04352004, 43.59181, 38.92629602, 38.42145994, 33.26062605, 30.60165444, 36.38823667, 13.44574942, 15.2183633, 14.90466815, 18.27718908, 24.88692394, 39.51729518, 26.1434299, 30.96870865, 29.54544363, 31.0876781, 38.25351179, 42.46729602, 43.71392551, 42.43810522, 40.35839054, 38.13177237, 31.6499585, 29.44868555, 20.51710989, 13.65037532, 13.82169328, 16.70465244, 17.59444988, 23.52986911, 26.31514239, 26.00239878, 28.36311771, 28.23168487, 29.66390558, 37.89042086, 44.07797817, 43.73687853, 42.15341324, 43.09722833, 38.74946239, 33.13134009, 31.5838583, 22.79193646]
right_hip_list_x = [29.31919987, 43.45575726, 44.37065722, 42.82309488, 39.55845987, 32.96600222, 27.00650114, 14.39395367, 12.98862431, 14.86849155, 16.96452666, 37.97908331, 23.75942536, 25.27192131, 29.90325398, 28.09775505, 33.46570328, 39.44761898, 43.64103253, 41.02275624, 18.39388163, 24.74277263, 27.01148472, 27.25124438, 24.78189465, 20.61128225, 11.18899556, 4.331458366, 8.701586318, 13.62376667, 19.15643239, 28.04187342, 24.84581497, 26.25428814, 30.3474104, 30.75532649, 38.645045, 41.60350366, 42.95684201, 42.12927613, 43.98263294, 24.53376653, 25.46072246, 25.66152509, 19.57991315, 11.21504768, 9.386064309, 7.631032122, 15.19960894, 19.89767739, 25.46647449, 25.53371193, 25.02838196, 26.51053326, 28.39498409, 33.82849743, 36.80274644, 42.73405214, 44.93049052, 42.45963591, 41.76081239, 39.78227981, 33.90976579, 25.4422636, 13.55829888, 7.014105696, 7.641504893, 10.16049929, 16.72544537, 20.88780231, 24.87005512, 26.06671961, 28.85557162, 27.18055218, 36.04502713, 38.53412478, 43.88971929, 39.30097867, 44.77052442, 39.56821083, 38.80058747, 33.38093144, 25.87647378, 15.41004466, 7.981988363, 7.378758114, 8.00610489, 12.53906485, 18.20585084, 24.19851543, 23.51879645, 25.45735419, 27.18902871, 33.56263087, 38.33780054, 42.70109481, 44.15946402, 41.89185507, 20.0796601, 22.96273914, 27.38250308, 22.26556171, 9.116317738, 3.939804418, 5.856869849, 8.756083978, 17.95305843, 20.69756366, 23.35179699, 23.52573998, 25.56005518, 26.23493586, 32.55229074, 38.50152467, 41.20592512, 44.42569199, 44.53153652, 41.27637739, 39.25378728, 32.31373336, 22.58086307, 14.37188546, 3.89661185, 1.327758897, 6.712838697, 13.75623988, 18.77511167, 23.64192066, 24.45357297, 26.26036693, 26.98211409, 32.29918233, 35.99781004, 43.94860963, 44.3206849, 44.79172921, 40.8742454, 20.74564971, 32.48881996, 23.99780186, 15.55512303, 8.167506956, 4.119343472, 7.617842366, 11.46828046, 19.74465704, 21.15367128, 23.70115823, 25.36159704, 24.61032016, 30.9811466, 35.11670405, 44.8610142, 12.99769315, 44.75193611, 41.30030855, 24.4258637, 35.91744245, 27.47479901, 22.61672978, 10.4346116, 6.252348051, 9.390341739, 15.91025396, 21.72528576, 22.83854367, 25.28609937, 25.48525749, 27.9866819, 20.00050548, 36.39713018, 41.79538279, 39.92037085, 40.57854167, 17.30373612, 21.1775441, 22.80903975, 32.18918796, 25.65912618, 13.5831374, 9.035956307, 8.073247191, 10.57328457, 18.28525569, 21.39112115, 27.60765197, 26.65197302, 28.79231539, 29.14159122, 12.57434063, 38.81704893, 44.3974774, 42.39899071, 43.82142928, 22.02920413, 24.58874246, 25.27319848, 21.04938976, 12.24303962, 6.725353039, 6.988126902, 9.98927221, 14.47134654, 21.0005811, 23.20663009, 24.84436894, 26.14185706, 19.59866533, 34.7184528, 37.68721499, 44.2782491, 41.38698406, 44.65815646, 22.24720813, 39.4265135, 30.43102529, 24.22905685, 14.45478614, 9.217737827, 7.417342046, 10.27241669, 16.10475818, 20.97172212, 24.43344623, 23.56251966, 27.57496466, 22.05410712, 34.00077276, 37.80318158, 6.015265663, 10.79161385, 44.03831432, 39.89394505, 26.42604827, 30.61825295, 23.2112047, 13.97969154, 8.714172968, 7.517277768, 10.26778374, 17.75770968, 21.20082648, 23.80936683, 24.99408458, 26.83587293, 27.6389691, 35.48126819, 38.67137447, 8.841070129, 38.33717785, 42.77966546, 44.4230469, 23.74746592, 28.69799382, 27.7512448, 20.47851377, 11.074608, 8.665263383, 7.766717842, 11.6587682, 19.143906, 23.27547332, 22.81844523, 26.25101514, 28.95992623, 32.08305862, 36.73181671, 43.6697354, 11.26127119, 42.34340364, 44.14123754, 21.71360312, 37.66400702, 27.49828919, 23.87325976, 10.29830973, 6.663514746, 41.73942406, 10.78568657, 16.27597146, 19.92784338, 22.2792373, 24.49819148, 26.73856676, 28.91839893, 36.48011118, 38.19148389, 44.66082382, 43.58552648, 44.01297057, 20.98264714, 25.9699624, 25.88701282, 23.03526861, 12.75901833, 7.732983469, 7.178909213, 11.66568705, 16.89394297, 24.6391363, 25.98067351, 27.79427847, 28.19181627, 26.90060556, 35.04202509, 40.51866187, 41.93878773, 12.99696864, 43.22656339, 22.8428515, 22.90744377, 27.2317631, 21.93592096, 13.78001126, 4.490433845, 8.031948472, 12.08635809, 14.77378172, 19.90790078, 24.70295921, 26.11345876, 26.83651593, 17.36645928, 33.17248862, 2.785987936, 10.5685746, 13.81025548, 43.39758872, 22.09320658, 26.454716, 30.76619759, 23.72863641, 15.37815603, 8.68792794, 8.09499638, 10.95783492, 18.57631303, 23.4369628, 24.88779535, 26.66875495, 29.71636898, 28.29811701, 34.22219119, 37.64392471, 10.28534245, 40.32056011, 41.85930507, 41.86120759, 39.13885754, 28.00394539, 25.48399328, 15.51312762, 7.592579683, 6.805952439, 9.734988794, 16.59063964, 20.55724961, 25.23174369, 25.24294507, 28.00155077, 25.73947196, 22.06501779, 36.35275207, 41.33591286, 42.61216914, 43.39269347, 42.17689603, 24.5432091, 27.163876, 27.68187507, 19.53663344, 12.51896916, 8.085682249, 10.24124018, 16.57898311, 17.86501466, 23.52092748, 24.6016328, 27.2835999, 28.71801433, 32.44474703, 36.07289303, 43.32719209, 42.29782982, 18.28132735, 44.23153734, 23.18109119, 26.76850947, 27.57930287, 19.74844877, 8.15026194, 7.29104402, 11.73505696, 13.27140776, 17.86895903, 24.05481962, 25.3969201, 28.43730625, 28.3567592, 18.90398106, 36.72676394, 42.00065422, 42.13732598, 42.99526393, 42.5523596, 23.61515135, 28.54434793, 28.90804037, 17.8571331, 9.042471075, 5.954261934, 10.77775941, 14.36951469, 17.52668375, 22.41889644, 25.01485331, 28.38481587, 26.87020684, 30.01829721, 33.77592056, 40.22272371, 43.79130565, 43.34459452, 40.56723224, 38.04345666, 26.71212135, 26.87170129, 20.97255781, 10.63734439, 5.974096862, 9.532047749, 15.17489622, 19.5776235, 22.49935096, 25.54883181, 30.14464984, 28.10350321, 33.18587165]
left_hip_rom = max(left_hip_list_x) - min(left_hip_list_x)
right_hip_rom = max(right_hip_list_x) - min(right_hip_list_x)

# CLASSIFY HIP DRIVE SCORES - LEFT HIP ROM
# 40 deg is good for slow running, more hip extension during faster running (50 deg flex to 10 deg hyperextension)
if left_hip_rom < 30 or left_hip_rom > 80:
    left_hip_norm = 10 # AWFUL
elif left_hip_rom > 30 and left_hip_rom < 40 or left_hip_rom > 70 and left_hip_rom < 80:
    left_hip_norm = 70 # AVERAGE
elif left_hip_rom > 40 and left_hip_rom < 50:
    left_hip_norm = 80 # GOOD
elif left_hip_rom > 50 and left_hip_rom < 60:
    left_hip_norm = 90 # EXCELLENT
elif left_hip_rom > 60 and left_hip_rom < 70:
    left_hip_norm = 99 # SUPERB

right_hip_list_x = [
    99.70, 98.94, 99.49, 99.23, 96.55, 93.73, 85.74, 80.14, 76.16, 66.73, 56.19, 47.02, 40.03, 28.25, 10.24, 14.28, 4.86,
    1.43, 10.52, 20.96, 37.45, 54.88, 66.91, 78.35, 82.37, 81.50, 85.32, 87.40, 99.11, 99.04, 95.40, 90.01, 86.87, 78.96,
    72.77, 62.59, 53.98, 47.25, 35.48, 23.26, 4.33, -0.70, -0.24, -0.28, 10.47, 27.50, 44.30, 65.92, 77.04, 81.44, 93.66,
    94.85, 96.36, 99.32, 100.44, 100.11, 99.36, 93.49, 85.53, 80.46, 74.79, 66.23, 53.37, 45.57, 38.27, 28.45, 18.28,
    8.50, 3.32, 5.20, 14.36, 27.87, 53.94, 59.98, 69.81, 78.68, 82.08, 90.89, 98.06, 98.56, 99.52, 99.86, 100.07, 98.44,
    95.80, 92.67, 83.10, 66.81, 62.96, 50.90, 40.43, 35.43, 25.40, 12.12, 5.64, 2.12, 1.17, 12.26, 29.46, 43.56, 53.09,
    73.54, 87.17, 88.47, 96.01, 97.06, 99.75, 100.84, 100.90, 100.71, 100.30, 95.28, 93.58, 85.66, 75.65, 69.51, 59.03,
    49.88, 41.70, 29.28, 17.79, 8.77, 7.78, 8.52, 19.82, 37.77, 44.19, 62.77, 72.34, 82.73, 83.97, 92.40, 97.03
]
# left_hip_list_x = right_hip_list_x + np.random.normal(0, 3.9, len(right_hip_list_x))
motion_hip = pd.DataFrame(
    {
        # "Left Hip": left_hip_list_x,
        "Hip": right_hip_list_x
    }
)

fig_hip = px.line(motion_hip, x=motion_hip.index/fs, y=["Hip"],
                  labels={"index": "Time (sec)"},
                  title="Motion of Hips",
                  width=800, height=400)

# fig_hip.update_layout(font=dict(size=24))
fig_hip.update_layout(
    xaxis_title="Time (sec)",
    yaxis_title="Angles (deg)",
    yaxis_title_font_size = 38, 
    xaxis_title_font_size = 38, 
    hoverlabel_font_size=38,
    title_font=dict(
        family="Courier New, monospace",
        size=40,
        color="white"
        ),
        xaxis=dict(
        tickfont=dict(
            size=28 
        ) 
        ),
        yaxis=dict(
        tickfont=dict(
        size=28 
        )
    ),

    legend=dict(
        title=dict(text=' ', font=dict(size=36)),  # Set legend title fontsize
        font=dict(size=32)  # Set legend label fontsize
    ))
st.plotly_chart(fig_hip, use_container_width=True)

chart_data2 = pd.DataFrame(
  {"Left Knee": left_knee, "Right Knee": right_knee})

chart_type = st.selectbox('Choose a chart type', ['Line', 'Bar']) 
## Create the chart
if chart_type == 'Line':
  fig = go.Figure()
  fig.add_trace(go.Line(y=chart_data2["Left Knee"].to_list(), x=np.arange(0, 100, 100/len(left_knee)),
                        mode='lines+markers',
                        name='Left Knee'))
  fig.add_trace(go.Line(y=chart_data2["Right Knee"].to_list(), x=np.arange(0, 100, 100/len(right_knee)),
                        mode='lines+markers',
                        name='Right Knee'))
  fig.update_layout(
        title="KNEE FLEXION ANGLES",
        xaxis_title="GAIT CYCLE (%)",
        yaxis_title="DEGREES",
        legend_font_size = 28,
        xaxis_title_font_size = 24, 
        yaxis_title_font_size = 24, 
        hoverlabel_font_size=24,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
        title_font=dict(
            # family="Courier New, monospace",
            size=42,
            color="white"
            ),
            xaxis=dict(
            tickfont=dict(size=36 
            ) 
            ),
            yaxis=dict(
            tickfont=dict(
            size=36 
        )
    )
    )
            
  st.plotly_chart(fig, use_container_width=True)

elif chart_type == 'Bar':
  st.bar_chart(chart_data2, y=["Left Knee", "Right Knee"])

# DIAL PLOTS  
dial1, dial2, dial3 = st.columns(3)
title_font_size = 26
with dial1:
  value = 90  # Value to be displayed on the dial (e.g., gas mileage)
  fig = go.Figure(go.Indicator(
      mode="gauge+number",
      value=value,
      domain={'x': [0, 1], 'y': [0, 1]},
      gauge=dict(
          axis=dict(range=[0, 100]),
          bar=dict(color="white"),
          borderwidth=2,
          bordercolor="gray",
          steps=[
              dict(range=[0, 25], color="red"),
              dict(range=[25, 50], color="orange"),
              dict(range=[50, 75], color="yellow"),
              dict(range=[75, 100], color="green")
          ],
          threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=value)
      )
  ))
  fig.update_layout(
      title={'text': "Hip Drive", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
      title_font_size = title_font_size,      
      font=dict(size=24)
  )
  st.plotly_chart(fig, use_container_width=True)
  # if hip drive is low, recommend hip mobility exercises & strengthening, if really low, also recommend arm swing exercises
  # recommended drills: SuperMarios, Hill Sprints, single leg hops, deadlifts
  st.write("## <div style='text-align: center;'><span style='color: green;'>GOOD</span>", unsafe_allow_html=True)

  with st.expander('Hip Drive'):
      st.write('Hip Drive is the power generated by your hips and glutes to propel you forward during running. Hip drive is important because it helps you run faster and more efficiently. A weak hip drive can lead to overstriding, which can lead to knee pain and shin splints. A strong hip drive can help you run faster and more efficiently.')
      url = "https://journals.biologists.com/jeb/article/215/11/1944/10883/Muscular-strategy-shift-in-human-running"
      st.link_button(":book: Read more about the importance of hip drive", url)
with dial2:
  value = 57 
  fig = go.Figure(go.Indicator(
      mode="gauge+number",
      value=value,
      domain={'x': [0, 1], 'y': [0, 1]},
      gauge=dict(
          axis=dict(range=[0, 100]),
          bar=dict(color="white"),
          borderwidth=2,
          bordercolor="gray",
          steps=[
              dict(range=[0, 25], color="red"),
              dict(range=[25, 50], color="orange"),
              dict(range=[50, 75], color="yellow"),
              dict(range=[75, 100], color="green")
          ],
          threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=value)
      )
  ))
  fig.update_layout(
      title={'text': "Foot Strike Score", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
      title_font_size = title_font_size,
      font=dict(size=24)
  )
  st.plotly_chart(fig, use_container_width=True)
  # if foot strike is low, recommend drills to increase cadence and reduce overstriding (e.g. high knees, butt kicks, Karaoke, and wind-sprints)
  st.write("## <div style='text-align: center;'><span style='color: red;'>POOR</span>", unsafe_allow_html=True)

  with st.expander("Foot Strike Score"):
      # st.plotly_chart(fig, use_container_width=True)
      st.write('Foot strike is the first point of contact between your foot and the ground. Foot strike should be on the midfoot, not the heel or the toes. If your foot strike is on your heel, it can lead to overstriding, which can lead to knee pain and shin splints. If your foot strike is on your toes, it can lead to calf pain and achilles tendonitis. A midfoot strike is ideal because it allows your foot to absorb the impact of the ground and propel you forward.')
      url2 ="https://journals.lww.com/nsca-jscr/abstract/2007/08000/foot_strike_patterns_of_runners_at_the_15_km_point.4"
      st.link_button(":book: Read more about the importance of foot strike", url2)
with dial3:
  value3 = 80  
  fig = go.Figure(go.Indicator(
      mode="gauge+number",
      value=value3,
      domain={'x': [0, 1], 'y': [0, 1]},
      gauge=dict(
          axis=dict(range=[0, 100]),
          bar=dict(color="white"),
          borderwidth=2,
          bordercolor="gray",
          steps=[
              dict(range=[0, 25], color="red"),
              dict(range=[25, 50], color="orange"),
              dict(range=[50, 75], color="yellow"),
              dict(range=[75, 100], color="green")
          ],
          threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=value3)
      )
  ))
  fig.update_layout(
      title={'text': "Arm Swing", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
      title_font_size = title_font_size,
      font=dict(size=24)
  )
  st.plotly_chart(fig, use_container_width=True)
  # if arm swing is low, then hip drive is low. Recommend hip mobility exercises and arm swing exercises
  st.write("## <div style='text-align: center;'><span style='color: green;'>GOOD</span>", unsafe_allow_html=True)

  with st.expander("Arm Swing"):
      # st.plotly_chart(fig, use_container_width=True)
      st.write('Arm Swing is important during running because it helps counterbalance the motion of the legs. Arm swing should not cross the midline of the body, but have more of a forward and back rocking motion. Arm swing helps your opposite leg drive forward during toe-off. A strong the arm-swing helps power your hips and knees to drive forward during running. A weak arm swing can lead to a weak hip drive and overstriding.')
      url = "https://journals.biologists.com/jeb/article/217/14/2456/12120/The-metabolic-cost-of-human-running-is-swinging"
      st.link_button(":book: Read more about the importance of arm swing", url)


#   # ----- UPLOAD AND RUN VIDEO FILE -----

# ---------- RUN VIDEO FILE --------------
from io import BytesIO
# url_squat = 'https://drive.google.com/uc?export=download&id=1OfosAFuI3UCs4TUqnxvrId4YqWjkPPwd'
url_squat = path + "depth_squat_instructions_transparent.png"

st.write("### Running Scores Determined from Range of Motion for slow and fast running from research by [Pink et al., 1994](%s)" % "https://journals.sagepub.com/doi/10.1177/036354659402200418?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed")
st.write("#### Upoad your video below :point_down:")
st.write("""###### Instructions for recording depth squat:
STEP 1: Position Setup""")
st.image(url_squat, caption="Depth Squat Instructions", width=500)
st.write("""
⦿ Record the participant from a 45-degree angle so you can see both the side and front of the participant
* Take 1-2 steps away and make sure the entire body is in the frame (including the feet)

STEP 2: Recording
         
⦿ Start the recording
* The participant should be standing with their feet shoulder width apart
* The participant should then squat down as far as they can go, just below parallel or 90 degrees
* The participant should then stand back up to the starting position
* The participant should repeat this 5 times
         
⦿ Stop the recording
         
STEP 3: Upload the video 
* Upload the video to the app
* Wait for the results to appear (this may take 2-3 minutes depending on how long your video is)        
""")

st.write("### Convert your MOV file to GIF using this software: https://cloudconvert.com/mov-to-gif")
uploaded_file = st.file_uploader("Choose an image...",  type=".gif") # change type=None to upload any file type (iphones use .MOV) 
if uploaded_file is not None:
    # Display a loading message
    start_time = time.time()
    progress_bar = st.progress(0)
    status_text = st.empty()
        
    # Simulate video processing (replace with actual processing code)
    for percent_complete in range(0, 101, 10):
        time.sleep(0.1)  # Simulating processing time
        progress_bar.progress(percent_complete)
        status_text.text(f"Processing: {percent_complete}%")

    # Display completion message
    st.success("Video processing complete!")
# with st.expander("Select a pre-recorded video"):
#   run_front_view = st.checkbox("Running- Front View")
#   run_side_view = st.checkbox("Running- Side View")
#   squat = st.checkbox("Squat")
#   single_leg_jump = st.checkbox("Single Leg Jump")
#   if run_front_view:
#       st.write(":runner:")
#       st.image(run_front, caption="Front View", width=300)
#   if run_side_view:
#       st.write(":bicyclist:")
#       st.image(run_side, caption="Front View", width=300)

#   if squat:
#       st.write(":weight_lifter:")
#       st.image(path + "depth_squat.gif", caption="Depth Squat", width=300)
#       uploaded_file = path + "depth_squat.gif"
#   if single_leg_jump:
#       st.image(path + "single_leg_jump.gif", caption = "Single Leg Jump", width=300)
#       uploaded_file = path + "single_leg_jump.gif"

# ======== MoveNet ========


import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import imageio
from PIL import Image
from IPython.display import HTML, display
# --- IMPORT MOVENET ---

module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
input_size = 192
def movenet(input_image):
  """Runs detection on an input image.

  Args:
    input_image: A [1, height, width, 3] tensor represents the input image
      pixels. Note that the height/width should already be resized and match the
      expected input resolution of the model before passing into this function.

  Returns:
    A [1, 1, 17, 3] float numpy array representing the predicted keypoint
    coordinates and scores.
  """
  model = module.signatures['serving_default']

  # SavedModel format expects tensor type of int32.
  input_image = tf.cast(input_image, dtype=tf.int32)
  # Run model inference.
  outputs = model(input_image)
  # Output is a [1, 1, 17, 3] tensor.
  keypoints_with_scores = outputs['output_0'].numpy()
  return keypoints_with_scores
# --------------------------------------
#@title Helper functions for visualization

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): '#E87722',  # Auburn Orange
    (0, 2): 'w',  # Navy Blue
    (1, 3): '#E87722',
    (2, 4): 'w',
    (0, 5): '#E87722',
    (0, 6): 'w',
    (5, 7): '#E87722',
    (7, 9): '#E87722',
    (6, 8): 'w',
    (8, 10): 'w',
    (5, 6): '#FFD100',  # Yellow
    (5, 11): '#E87722',
    (6, 12): 'w',
    (11, 12): '#FFD100',
    (11, 13): '#E87722',
    (13, 15): '#E87722',
    (12, 14): 'w',
    (14, 16): 'w'
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=100, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    # image_from_plot = cv2.resize(
    #     image_from_plot, dsize=(output_image_width, output_image_height),
    #      interpolation=cv2.INTER_CUBIC)
    image_from_plot = Image.fromarray(image_from_plot)
    image_from_plot = image_from_plot.resize(
    (output_image_width, output_image_height), resample=Image.Resampling.LANCZOS)
    image_from_plot = np.array(image_from_plot)
    
  return image_from_plot

def to_gif(images, duration):
  """Converts image sequence (4D numpy array) to gif."""
  imageio.mimsave('./animation.gif', images, duration=duration)
  return embed.embed_file('./animation.gif')

def progress(value, max=100):
  return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max))

# ----
#@title Cropping Algorithm

# Confidence score to determine whether a keypoint prediction is reliable.
MIN_CROP_KEYPOINT_SCORE = 0.25

def init_crop_region(image_height, image_width):
  """Defines the default crop region.

  The function provides the initial crop region (pads the full image from both
  sides to make it a square image) when the algorithm cannot reliably determine
  the crop region from the previous frame.
  """
  if image_width > image_height:
    box_height = image_width / image_height
    box_width = 1.0
    y_min = (image_height / 2 - image_width / 2) / image_height
    x_min = 0.0
  else:
    box_height = 1.0
    box_width = image_height / image_width
    y_min = 0.0
    x_min = (image_width / 2 - image_height / 2) / image_width

  return {
    'y_min': y_min,
    'x_min': x_min,
    'y_max': y_min + box_height,
    'x_max': x_min + box_width,
    'height': box_height,
    'width': box_width
  }

def torso_visible(keypoints):
  """Checks whether there are enough torso keypoints.

  This function checks whether the model is confident at predicting one of the
  shoulders/hips which is required to determine a good crop region.
  """
  return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE) and
          (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE))

def determine_torso_and_body_range(
    keypoints, target_keypoints, center_y, center_x):
  """Calculates the maximum distance from each keypoints to the center location.

  The function returns the maximum distances from the two sets of keypoints:
  full 17 keypoints and 4 torso keypoints. The returned information will be
  used to determine the crop size. See determineCropRegion for more detail.
  """
  torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
  max_torso_yrange = 0.0
  max_torso_xrange = 0.0
  for joint in torso_joints:
    dist_y = abs(center_y - target_keypoints[joint][0])
    dist_x = abs(center_x - target_keypoints[joint][1])
    if dist_y > max_torso_yrange:
      max_torso_yrange = dist_y
    if dist_x > max_torso_xrange:
      max_torso_xrange = dist_x

  max_body_yrange = 0.0
  max_body_xrange = 0.0
  for joint in KEYPOINT_DICT.keys():
    if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
      continue
    dist_y = abs(center_y - target_keypoints[joint][0]);
    dist_x = abs(center_x - target_keypoints[joint][1]);
    if dist_y > max_body_yrange:
      max_body_yrange = dist_y

    if dist_x > max_body_xrange:
      max_body_xrange = dist_x

  return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

def determine_crop_region(
      keypoints, image_height,
      image_width):
  """Determines the region to crop the image for the model to run inference on.

  The algorithm uses the detected joints from the previous frame to estimate
  the square region that encloses the full body of the target person and
  centers at the midpoint of two hip joints. The crop size is determined by
  the distances between each joints and the center point.
  When the model is not confident with the four torso joint predictions, the
  function returns a default crop which is the full image padded to square.
  """
  target_keypoints = {}
  for joint in KEYPOINT_DICT.keys():
    target_keypoints[joint] = [
      keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
      keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
    ]

  if torso_visible(keypoints):
    center_y = (target_keypoints['left_hip'][0] +
                target_keypoints['right_hip'][0]) / 2;
    center_x = (target_keypoints['left_hip'][1] +
                target_keypoints['right_hip'][1]) / 2;

    (max_torso_yrange, max_torso_xrange,
      max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
          keypoints, target_keypoints, center_y, center_x)

    crop_length_half = np.amax(
        [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
          max_body_yrange * 1.2, max_body_xrange * 1.2])

    tmp = np.array(
        [center_x, image_width - center_x, center_y, image_height - center_y])
    crop_length_half = np.amin(
        [crop_length_half, np.amax(tmp)]);

    crop_corner = [center_y - crop_length_half, center_x - crop_length_half];

    if crop_length_half > max(image_width, image_height) / 2:
      return init_crop_region(image_height, image_width)
    else:
      crop_length = crop_length_half * 2;
      return {
        'y_min': crop_corner[0] / image_height,
        'x_min': crop_corner[1] / image_width,
        'y_max': (crop_corner[0] + crop_length) / image_height,
        'x_max': (crop_corner[1] + crop_length) / image_width,
        'height': (crop_corner[0] + crop_length) / image_height -
            crop_corner[0] / image_height,
        'width': (crop_corner[1] + crop_length) / image_width -
            crop_corner[1] / image_width
      }
  else:
    return init_crop_region(image_height, image_width)

def crop_and_resize(image, crop_region, crop_size):
  """Crops and resize the image to prepare for the model input."""
  boxes=[[crop_region['y_min'], crop_region['x_min'],
          crop_region['y_max'], crop_region['x_max']]]
  output_image = tf.image.crop_and_resize(
      image, box_indices=[0], boxes=boxes, crop_size=crop_size)
  return output_image

def run_inference(movenet, image, crop_region, crop_size):
  """Runs model inferece on the cropped region.

  The function runs the model inference on the cropped region and updates the
  model output to the original image coordinate system.
  """
  image_height, image_width, _ = image.shape
  input_image = crop_and_resize(
    tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
  # Run model inference.
  keypoints_with_scores = movenet(input_image)
  # Update the coordinates.
  for idx in range(17):
    keypoints_with_scores[0, 0, idx, 0] = (
        crop_region['y_min'] * image_height +
        crop_region['height'] * image_height *
        keypoints_with_scores[0, 0, idx, 0]) / image_height
    keypoints_with_scores[0, 0, idx, 1] = (
        crop_region['x_min'] * image_width +
        crop_region['width'] * image_width *
        keypoints_with_scores[0, 0, idx, 1]) / image_width
  return keypoints_with_scores


# uploaded_file = "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/baseline_pics/run_treadmill_outdoors_cut1.gif"
if uploaded_file is not None:
  # update for .MOV  ========= START =========
  # import moviepy
  # from moviepy.editor import VideoFileClip
  # file_name = uploaded_file.name
  # file_path = "/workspaces/PolarPlotter/baseline_pics/" + str(file_name)
  # st.write(file_path)

  # with open(file_path, "wb") as f:
  #     f.write(uploaded_file.read())
  # st.success(f"File saved to: {file_path}")
  # path2mov = r"/workspaces/PolarPlotter/baseline_pics/" + str(file_name) 
  # gif_file = path2mov[:-4] + '.gif'
  # videoClip = moviepy.editor.VideoFileClip(path2mov)
  # videoClip.write_gif(gif_file)
  # image_content = tf.io.read_file(gif_file)

  # image = tf.io.read_file(gif_file)
  # image = tf.image.decode_gif(image)
  # num_frames, image_height, image_width, _ = image.shape
  # st.write(num_frames, image_height, image_width)
  # num_frames=115
  # update for .MOV  ========= END=======

  image_content = uploaded_file.read()
  image = tf.image.decode_gif(image_content)
  num_frames, image_height, image_width, _ = image.shape
  crop_region = init_crop_region(image_height, image_width)

  nose_list_x, left_shoulder_list_x, right_shoulder_list_x,left_elbow_list_x, right_elbow_list_x, left_wrist_list_x, right_wrist_list_x = [],[],[],[],[],[],[]
  left_ankle_list_x, right_ankle_list_x, left_hip_list_x, right_hip_list_x, left_knee_list_x, right_knee_list_x = [], [], [], [], [] ,[]

  nose_list_y, left_shoulder_list_y, right_shoulder_list_y,left_elbow_list_y, right_elbow_list_y, left_wrist_list_y, right_wrist_list_y = [],[],[],[],[],[],[]
  left_ankle_list_y, right_ankle_list_y, left_hip_list_y, right_hip_list_y, left_knee_list_y, right_knee_list_y = [], [], [], [], [] ,[]

  nose_list_conf, left_shoulder_list_conf, right_shoulder_list_conf,left_elbow_list_conf, right_elbow_list_conf, left_wrist_list_conf, right_wrist_list_conf = [],[],[],[],[],[],[]
  left_ankle_list_conf, right_ankle_list_conf, left_hip_list_conf, right_hip_list_conf, left_knee_list_conf, right_knee_list_conf = [], [], [], [], [] ,[]

  left_knee_angle_list_x, right_knee_angle_list_x = [], []
  left_knee_deg_list, right_knee_deg_list = [], []
  output_images = []
  bar = display(progress(0, image.shape[0]-1), display_id=True)
  for frame_idx in range(num_frames):
    keypoints_with_scores = run_inference(
        movenet, image[frame_idx, :, :, :], crop_region,
        crop_size=[input_size, input_size])
    output_images.append(draw_prediction_on_image(
        image[frame_idx, :, :, :].numpy().astype(np.int32),
        keypoints_with_scores, crop_region=None,
        close_figure=True, output_image_height=300))
    crop_region = determine_crop_region(
        keypoints_with_scores, image_height, image_width)
    # bar.update(progress(frame_idx, num_frames-1))

    nose_x = keypoints_with_scores[0,0,0,0]
    left_shoulder_x = keypoints_with_scores[0,0,5,0]
    right_shoulder_x = keypoints_with_scores[0,0,6,0]
    left_elbow_x = keypoints_with_scores[0,0,7,0]
    right_elbow_x = keypoints_with_scores[0,0,8,0]
    left_wrist_x = keypoints_with_scores[0,0,9,0]
    right_wrist_x = keypoints_with_scores[0,0,10,0]
    # HIPS
    left_hip_x = keypoints_with_scores[0,0,11,0]
    right_hip_x = keypoints_with_scores[0,0,12,0]
    # ANKLES
    left_ankle_x = keypoints_with_scores[0,0,15,0]
    right_ankle_x = keypoints_with_scores[0,0,16,0]
    # KNEES
    left_knee_x = keypoints_with_scores[0,0,13,0]
    right_knee_x = keypoints_with_scores[0,0,14,0]

    # Append keypoints to list
    nose_list_x.append(nose_x)
    left_shoulder_list_x.append(left_shoulder_x)
    right_shoulder_list_x.append(right_shoulder_x)
    left_elbow_list_x.append(left_elbow_x)
    right_elbow_list_x.append(right_elbow_x)
    left_wrist_list_x.append(left_wrist_x)
    right_wrist_list_x.append(right_wrist_x)
    left_ankle_list_x.append(left_ankle_x)
    right_ankle_list_x.append(right_ankle_x)
    left_knee_list_x.append(left_knee_x)
    right_knee_list_x.append(right_knee_x)
    left_hip_list_x.append(left_hip_x)
    right_hip_list_x.append(right_hip_x)


    nose_y = keypoints_with_scores[0,0,0,1]
    left_shoulder_y = keypoints_with_scores[0,0,5,1]
    right_shoulder_y = keypoints_with_scores[0,0,6,1]
    left_elbow_y = keypoints_with_scores[0,0,7,1]
    right_elbow_y = keypoints_with_scores[0,0,8,1]
    left_wrist_y = keypoints_with_scores[0,0,9,1]
    right_wrist_y = keypoints_with_scores[0,0,10,1]
    # HIPS
    left_hip_y = keypoints_with_scores[0,0,11,1]
    right_hip_y = keypoints_with_scores[0,0,12,1]
    # ANKLES
    left_ankle_y = keypoints_with_scores[0,0,15,1]
    right_ankle_y = keypoints_with_scores[0,0,16,1]
    # KNEES
    left_knee_y = keypoints_with_scores[0,0,13,1]
    right_knee_y = keypoints_with_scores[0,0,14,1]

    # Append keypoints to list
    nose_list_y.append(nose_y)
    left_shoulder_list_y.append(left_shoulder_y)
    right_shoulder_list_y.append(right_shoulder_y)
    left_elbow_list_y.append(left_elbow_y)
    right_elbow_list_y.append(right_elbow_y)
    left_wrist_list_y.append(left_wrist_y)
    right_wrist_list_y.append(right_wrist_y)
    left_ankle_list_y.append(left_ankle_y)
    right_ankle_list_y.append(right_ankle_y)
    left_knee_list_y.append(left_knee_y)
    right_knee_list_y.append(right_knee_y)
    left_hip_list_y.append(left_hip_y)
    right_hip_list_y.append(right_hip_y)

    nose_c = keypoints_with_scores[0,0,0,2]
    left_shoulder_c = keypoints_with_scores[0,0,5,2]
    right_shoulder_c = keypoints_with_scores[0,0,6,2]
    left_elbow_c = keypoints_with_scores[0,0,7,2]
    right_elbow_c = keypoints_with_scores[0,0,8,2]
    left_wrist_c = keypoints_with_scores[0,0,9,2]
    right_wrist_c = keypoints_with_scores[0,0,10,2]
    # HIPS
    left_hip_c = keypoints_with_scores[0,0,11,2]
    right_hip_c = keypoints_with_scores[0,0,12,2]
    # ANKLES
    left_ankle_c = keypoints_with_scores[0,0,15,2]
    right_ankle_c = keypoints_with_scores[0,0,16,2]
    # KNEES
    left_knee_c = keypoints_with_scores[0,0,13,2]
    right_knee_c = keypoints_with_scores[0,0,14,2]

    # Append keypoints to list
    nose_list_conf.append(nose_c)
    left_shoulder_list_conf.append(left_shoulder_c)
    right_shoulder_list_conf.append(right_shoulder_c)
    left_elbow_list_conf.append(left_elbow_c)
    right_elbow_list_conf.append(right_elbow_c)
    left_wrist_list_conf.append(left_wrist_c)
    right_wrist_list_conf.append(right_wrist_c)
    left_ankle_list_conf.append(left_ankle_c)
    right_ankle_list_conf.append(right_ankle_c)
    left_knee_list_conf.append(left_knee_c)
    right_knee_list_conf.append(right_knee_c)
    left_hip_list_conf.append(left_hip_c)
    right_hip_list_conf.append(right_hip_c)
    
  output = np.stack(output_images, axis=0)
  image_capture = to_gif(output, duration=100)

  def euclidean_distance(array):
    euclidean_distance = np.linalg.norm(array)
    return euclidean_distance

  # get the euclidean distance of the medio-lateral directions
  left_knee_norm = euclidean_distance(left_knee_list_y) / num_frames
  right_knee_norm = euclidean_distance(right_knee_list_y) / num_frames
  left_hip_norm = euclidean_distance(left_hip_list_y) / num_frames
  right_hip_norm = euclidean_distance(right_hip_list_y) / num_frames
  left_shoulder_norm = euclidean_distance(left_shoulder_list_y) / num_frames
  right_shoulder_norm = euclidean_distance(right_shoulder_list_y) / num_frames
  st.write(image_capture)

  # TOTAL TIME TO RUN
    # Calculate the elapsed time
  elapsed_time = time.time() - start_time

  # Display the elapsed time
  st.success(f"Video processing completed in {elapsed_time:.2f} seconds.")
      

  """
  ## Video Results
  """
  fs=25
  # Function to plot the data
  # def plot_results(left_knee_norm, right_knee_norm, left_hip_norm, right_hip_norm,
  #                  left_hip_list_x, right_hip_list_x, left_knee_list_x, right_knee_list_x):

  # PLOT CONFIDENCE SCORES
  df = pd.DataFrame(
    {
        "Joint": ['Nose', 'L Wrist', 'R Wrist', 'L Shoulder', 'R Shoulder', 'L Ankle', 'R Ankle',
                  'L Hip', 'R Hip', 'L Knee', 'R Knee'],
        "Confidence Score": [nose_list_conf, left_wrist_list_conf, right_wrist_list_conf, left_shoulder_list_conf, right_shoulder_list_conf, left_ankle_list_conf, right_ankle_list_conf, left_hip_list_conf, right_hip_list_conf, left_knee_list_conf, right_knee_list_conf],
    }
  )  
  df_unstacked = df.set_index('Joint')['Confidence Score'].apply(pd.Series).stack().reset_index()
  df_unstacked.columns = ['Joint', 'Index', 'Confidence Score']

  # Plotly Boxplot
  fig_conf = px.box(df_unstacked, x='Joint', y='Confidence Score', points="all", title="Confidence Score Boxplots",
              labels={"Confidence Score": "Confidence Score", "Joint": "Joint"})

  fig_conf.update_layout(
        xaxis_title="",
        yaxis_title="",
        yaxis_title_font_size = 38, 
        xaxis_title_font_size = 38, 
        hoverlabel_font_size=12,
        title_font=dict(
            family="Courier New, monospace",
            size=32,
            color="white"
            ),
            xaxis=dict(
            tickfont=dict(
                size=24 
            ) 
            ),
            yaxis=dict(
            tickfont=dict(
            size=32 
        ),
        
    ))
  st.plotly_chart(fig_conf, use_container_width=True)

  # PLOT STABILITY SCORES
  chart_data = pd.DataFrame(
      {
          "Joint": ['Left Knee', 'Right Knee', 'Left Hip', 'Right Hip'],
          "Stability Score": [left_knee_norm, right_knee_norm, left_hip_norm, right_hip_norm],
      }
  )

  fig_bar = px.bar(chart_data, x="Joint", y="Stability Score", color="Joint",
                  labels={"Stability Score": "Stability Score"},
                  title="Stability Score by Joint",
                  width=600, height=400)

  fig_bar.update_layout(
      xaxis_title="",
      yaxis_title="",
      yaxis_title_font_size = 38, 
      xaxis_title_font_size = 38, 
      hoverlabel_font_size=38,
      title_font=dict(
          family="Courier New, monospace",
          size=36,
          color="white"
          ),
          xaxis=dict(
          tickfont=dict(
              size=28 
          ) 
          ),
          yaxis=dict(
          tickfont=dict(
          size=28 
      )
      ),
          legend=dict(
          title=dict(text='Joint', font=dict(size=36)),  
          font=dict(size=32) 
  )
  )
  st.plotly_chart(fig_bar, use_container_width=True)

  # left_hip_list_x = [33.53389631, 23.60319908, 24.71082991, 31.44837908, 35.71872569, 32.96516151, 28.28836734, 40.9953792, 39.86062259, 33.44414823, 41.47076439, 25.85460365, 40.20734433, 43.9932253, 34.5009277, 29.32044717, 29.53059614, 21.10850017, 13.16118136, 42.41039559, 42.95774915, 41.49441895, 43.0807489, 38.58519925, 30.44002691, 28.2239258, 40.58501604, 41.97357465, 44.90092795, 37.47970917, 35.40727071, 42.80065084, 39.28141527, 30.62938808, 30.86638045, 21.34408458, 14.03436804, 14.80872355, 15.6054373, 20.89026544, 27.79409606, 42.84784067, 34.18172167, 27.93136592, 23.0611644, 34.7116706, 39.66439584, 38.32268776, 40.95874529, 39.52790611, 40.3492191, 43.99573291, 34.52372057, 26.97184025, 19.75252846, 14.5810455, 12.50459463, 17.68137021, 19.12518642, 23.35677677, 26.80732765, 22.53992096, 32.53392676, 29.00108124, 32.64459898, 36.26132052, 42.94534961, 43.66988716, 38.93033199, 44.78470186, 44.11693197, 42.60460079, 30.99932869, 28.07656634, 18.48125998, 11.7676165, 10.50151725, 14.30195246, 19.98869178, 23.24740766, 22.16055184, 31.04804809, 28.28108404, 29.54036391, 37.26289185, 43.88307393, 43.81017342, 42.58629115, 44.58570067, 41.57446955, 34.67608359, 30.48367938, 26.06455147, 15.29782241, 14.43897893, 15.45686709, 16.54996568, 23.87839774, 21.79575576, 34.84248192, 30.02331517, 28.69853475, 30.78808398, 35.39675975, 44.30543326, 42.28800579, 38.62323853, 43.17507745, 43.91494867, 33.62113186, 31.7366487, 29.13153056, 13.79078316, 7.864036973, 18.61875874, 18.12726589, 24.03468475, 28.42438116, 29.78725237, 32.69282359, 26.41422407, 34.29385687, 36.39771945, 42.15621193, 44.21451828, 39.64657897, 44.13566782, 43.32370826, 41.11079839, 32.92536482, 30.74720525, 19.02300658, 13.69007524, 6.812642954, 12.78933252, 15.73457867, 20.35307427, 39.12753915, 30.28039795, 27.01337101, 30.26136371, 32.32609192, 37.26667204, 44.21785092, 43.84530357, 40.36223294, 41.52678493, 40.78510003, 32.66838017, 27.88958285, 21.50134539, 13.99564581, 9.11447025, 44.82410269, 20.1371615, 25.20922098, 40.69251027, 29.75513652, 31.98474818, 29.00873643, 33.22456161, 39.11029684, 41.43555511, 39.47963554, 35.61925248, 42.61342776, 42.68236169, 33.74263833, 32.08062399, 32.18553246, 15.06109906, 11.07342806, 4.995704638, 14.04262182, 44.68124311, 42.45264572, 21.99539729, 32.07216013, 29.69709639, 31.63088262, 34.78498766, 44.3200169, 42.18827131, 42.89356415, 41.5333971, 40.95663346, 36.49648729, 30.93758773, 31.95502352, 36.39595956, 15.78215014, 14.45672484, 20.15271369, 21.58385902, 41.96448377, 32.91886636, 25.7161234, 25.97693594, 32.45653516, 36.76897468, 37.7840693, 44.76747923, 43.64363636, 44.11041891, 41.46010352, 33.43164205, 30.33623782, 32.80668392, 17.38250782, 12.87153597, 12.39614202, 13.39012073, 20.64395774, 42.83184903, 28.28767506, 30.79510535, 30.47389001, 31.54525183, 36.4800443, 40.94045451, 42.46226744, 43.08296369, 44.34570493, 40.71319723, 31.20813476, 28.95069164, 28.75597537, 16.16694179, 15.11659614, 42.79760516, 41.58047058, 21.18260715, 27.7461244, 34.92290118, 30.56732981, 25.52230264, 27.33104502, 35.60753191, 40.37660966, 43.86999536, 42.82397883, 40.75664128, 41.81206516, 33.87176183, 30.86724373, 27.42799601, 16.74048485, 11.23379157, 44.00928312, 7.751710741, 21.6357455, 20.69741986, 21.93927661, 35.38201634, 32.47691507, 31.98132085, 33.73964788, 39.85018041, 41.30632962, 44.34456173, 43.31054297, 41.8082906, 38.35660249, 31.70020241, 31.13296975, 21.99764724, 13.98269421, 11.02302375, 43.12316318, 17.56531887, 24.52042158, 39.03335609, 27.32608782, 28.16847002, 28.18249308, 34.79935298, 37.53402797, 8.426470828, 41.50707735, 44.35685029, 40.15235281, 37.23373901, 29.09625425, 28.0897296, 26.46027802, 16.71007606, 16.68649847, 17.63270282, 20.28153719, 17.49101975, 39.97306823, 33.25426775, 29.25109943, 28.76111863, 31.4884906, 32.83496515, 38.8388699, 44.96944214, 44.16184395, 41.78403062, 41.51636011, 37.4168382, 34.46849445, 28.86623935, 21.98216784, 14.59481254, 6.14812992, 44.76069516, 18.30746555, 39.34292787, 35.58966561, 26.57119627, 27.02718321, 30.12508136, 34.89009772, 35.66234887, 40.10961257, 42.31393165, 40.20681846, 42.2393426, 30.31177974, 27.82691954, 33.69458137, 15.12538954, 4.178903579, 44.92359197, 40.56550992, 25.32026441, 21.89884248, 36.82974855, 31.27478807, 27.30814426, 32.20596541, 34.39741846, 37.95954392, 44.99068749, 42.87867902, 42.66811125, 34.57827368, 34.493955, 32.72163913, 33.64816712, 16.41584451, 11.37893962, 43.77401781, 12.1107742, 15.51938538, 23.42276476, 22.2262684, 26.54283276, 27.17153831, 33.31046306, 35.21733044, 34.21020358, 44.75615839, 42.7055645, 41.28503993, 38.04523472, 38.24964355, 30.28766225, 28.5044385, 31.19175074, 14.7182323, 16.72894939, 16.44930339, 21.6849663, 23.10101859, 40.1433722, 28.20050926, 30.57031842, 31.65182871, 32.04789844, 41.30662656, 42.09292162, 43.50567943, 40.0949953, 40.83884384, 33.30515478, 29.70954083, 27.6663512, 21.45267022, 14.63905549, 10.3853176, 16.06787457, 43.66462563, 20.84927058, 22.09729842, 29.11657631, 29.31671525, 30.68366658, 30.13373694, 37.034661, 41.06764124, 43.04352004, 43.59181, 38.92629602, 38.42145994, 33.26062605, 30.60165444, 36.38823667, 13.44574942, 15.2183633, 14.90466815, 18.27718908, 24.88692394, 39.51729518, 26.1434299, 30.96870865, 29.54544363, 31.0876781, 38.25351179, 42.46729602, 43.71392551, 42.43810522, 40.35839054, 38.13177237, 31.6499585, 29.44868555, 20.51710989, 13.65037532, 13.82169328, 16.70465244, 17.59444988, 23.52986911, 26.31514239, 26.00239878, 28.36311771, 28.23168487, 29.66390558, 37.89042086, 44.07797817, 43.73687853, 42.15341324, 43.09722833, 38.74946239, 33.13134009, 31.5838583, 22.79193646]
  # right_hip_list_x = [29.31919987, 43.45575726, 44.37065722, 42.82309488, 39.55845987, 32.96600222, 27.00650114, 14.39395367, 12.98862431, 14.86849155, 16.96452666, 37.97908331, 23.75942536, 25.27192131, 29.90325398, 28.09775505, 33.46570328, 39.44761898, 43.64103253, 41.02275624, 18.39388163, 24.74277263, 27.01148472, 27.25124438, 24.78189465, 20.61128225, 11.18899556, 4.331458366, 8.701586318, 13.62376667, 19.15643239, 28.04187342, 24.84581497, 26.25428814, 30.3474104, 30.75532649, 38.645045, 41.60350366, 42.95684201, 42.12927613, 43.98263294, 24.53376653, 25.46072246, 25.66152509, 19.57991315, 11.21504768, 9.386064309, 7.631032122, 15.19960894, 19.89767739, 25.46647449, 25.53371193, 25.02838196, 26.51053326, 28.39498409, 33.82849743, 36.80274644, 42.73405214, 44.93049052, 42.45963591, 41.76081239, 39.78227981, 33.90976579, 25.4422636, 13.55829888, 7.014105696, 7.641504893, 10.16049929, 16.72544537, 20.88780231, 24.87005512, 26.06671961, 28.85557162, 27.18055218, 36.04502713, 38.53412478, 43.88971929, 39.30097867, 44.77052442, 39.56821083, 38.80058747, 33.38093144, 25.87647378, 15.41004466, 7.981988363, 7.378758114, 8.00610489, 12.53906485, 18.20585084, 24.19851543, 23.51879645, 25.45735419, 27.18902871, 33.56263087, 38.33780054, 42.70109481, 44.15946402, 41.89185507, 20.0796601, 22.96273914, 27.38250308, 22.26556171, 9.116317738, 3.939804418, 5.856869849, 8.756083978, 17.95305843, 20.69756366, 23.35179699, 23.52573998, 25.56005518, 26.23493586, 32.55229074, 38.50152467, 41.20592512, 44.42569199, 44.53153652, 41.27637739, 39.25378728, 32.31373336, 22.58086307, 14.37188546, 3.89661185, 1.327758897, 6.712838697, 13.75623988, 18.77511167, 23.64192066, 24.45357297, 26.26036693, 26.98211409, 32.29918233, 35.99781004, 43.94860963, 44.3206849, 44.79172921, 40.8742454, 20.74564971, 32.48881996, 23.99780186, 15.55512303, 8.167506956, 4.119343472, 7.617842366, 11.46828046, 19.74465704, 21.15367128, 23.70115823, 25.36159704, 24.61032016, 30.9811466, 35.11670405, 44.8610142, 12.99769315, 44.75193611, 41.30030855, 24.4258637, 35.91744245, 27.47479901, 22.61672978, 10.4346116, 6.252348051, 9.390341739, 15.91025396, 21.72528576, 22.83854367, 25.28609937, 25.48525749, 27.9866819, 20.00050548, 36.39713018, 41.79538279, 39.92037085, 40.57854167, 17.30373612, 21.1775441, 22.80903975, 32.18918796, 25.65912618, 13.5831374, 9.035956307, 8.073247191, 10.57328457, 18.28525569, 21.39112115, 27.60765197, 26.65197302, 28.79231539, 29.14159122, 12.57434063, 38.81704893, 44.3974774, 42.39899071, 43.82142928, 22.02920413, 24.58874246, 25.27319848, 21.04938976, 12.24303962, 6.725353039, 6.988126902, 9.98927221, 14.47134654, 21.0005811, 23.20663009, 24.84436894, 26.14185706, 19.59866533, 34.7184528, 37.68721499, 44.2782491, 41.38698406, 44.65815646, 22.24720813, 39.4265135, 30.43102529, 24.22905685, 14.45478614, 9.217737827, 7.417342046, 10.27241669, 16.10475818, 20.97172212, 24.43344623, 23.56251966, 27.57496466, 22.05410712, 34.00077276, 37.80318158, 6.015265663, 10.79161385, 44.03831432, 39.89394505, 26.42604827, 30.61825295, 23.2112047, 13.97969154, 8.714172968, 7.517277768, 10.26778374, 17.75770968, 21.20082648, 23.80936683, 24.99408458, 26.83587293, 27.6389691, 35.48126819, 38.67137447, 8.841070129, 38.33717785, 42.77966546, 44.4230469, 23.74746592, 28.69799382, 27.7512448, 20.47851377, 11.074608, 8.665263383, 7.766717842, 11.6587682, 19.143906, 23.27547332, 22.81844523, 26.25101514, 28.95992623, 32.08305862, 36.73181671, 43.6697354, 11.26127119, 42.34340364, 44.14123754, 21.71360312, 37.66400702, 27.49828919, 23.87325976, 10.29830973, 6.663514746, 41.73942406, 10.78568657, 16.27597146, 19.92784338, 22.2792373, 24.49819148, 26.73856676, 28.91839893, 36.48011118, 38.19148389, 44.66082382, 43.58552648, 44.01297057, 20.98264714, 25.9699624, 25.88701282, 23.03526861, 12.75901833, 7.732983469, 7.178909213, 11.66568705, 16.89394297, 24.6391363, 25.98067351, 27.79427847, 28.19181627, 26.90060556, 35.04202509, 40.51866187, 41.93878773, 12.99696864, 43.22656339, 22.8428515, 22.90744377, 27.2317631, 21.93592096, 13.78001126, 4.490433845, 8.031948472, 12.08635809, 14.77378172, 19.90790078, 24.70295921, 26.11345876, 26.83651593, 17.36645928, 33.17248862, 2.785987936, 10.5685746, 13.81025548, 43.39758872, 22.09320658, 26.454716, 30.76619759, 23.72863641, 15.37815603, 8.68792794, 8.09499638, 10.95783492, 18.57631303, 23.4369628, 24.88779535, 26.66875495, 29.71636898, 28.29811701, 34.22219119, 37.64392471, 10.28534245, 40.32056011, 41.85930507, 41.86120759, 39.13885754, 28.00394539, 25.48399328, 15.51312762, 7.592579683, 6.805952439, 9.734988794, 16.59063964, 20.55724961, 25.23174369, 25.24294507, 28.00155077, 25.73947196, 22.06501779, 36.35275207, 41.33591286, 42.61216914, 43.39269347, 42.17689603, 24.5432091, 27.163876, 27.68187507, 19.53663344, 12.51896916, 8.085682249, 10.24124018, 16.57898311, 17.86501466, 23.52092748, 24.6016328, 27.2835999, 28.71801433, 32.44474703, 36.07289303, 43.32719209, 42.29782982, 18.28132735, 44.23153734, 23.18109119, 26.76850947, 27.57930287, 19.74844877, 8.15026194, 7.29104402, 11.73505696, 13.27140776, 17.86895903, 24.05481962, 25.3969201, 28.43730625, 28.3567592, 18.90398106, 36.72676394, 42.00065422, 42.13732598, 42.99526393, 42.5523596, 23.61515135, 28.54434793, 28.90804037, 17.8571331, 9.042471075, 5.954261934, 10.77775941, 14.36951469, 17.52668375, 22.41889644, 25.01485331, 28.38481587, 26.87020684, 30.01829721, 33.77592056, 40.22272371, 43.79130565, 43.34459452, 40.56723224, 38.04345666, 26.71212135, 26.87170129, 20.97255781, 10.63734439, 5.974096862, 9.532047749, 15.17489622, 19.5776235, 22.49935096, 25.54883181, 30.14464984, 28.10350321, 33.18587165]
  left_hip_rom = max(left_hip_list_x) - min(left_hip_list_x)
  right_hip_rom = max(right_hip_list_x) - min(right_hip_list_x)

  # CLASSIFY HIP DRIVE SCORES - LEFT HIP ROM
  if left_hip_rom < 30 or left_hip_rom > 80:
      left_hip_norm = 10 # AWFUL
  elif left_hip_rom > 30 and left_hip_rom < 40 or left_hip_rom > 70 and left_hip_rom < 80:
      left_hip_norm = 70 # AVERAGE
  elif left_hip_rom > 40 and left_hip_rom < 50:
      left_hip_norm = 80 # GOOD
  elif left_hip_rom > 50 and left_hip_rom < 60:
      left_hip_norm = 90 # EXCELLENT
  elif left_hip_rom > 60 and left_hip_rom < 70:
      left_hip_norm = 99 # SUPERB

  motion_hip = pd.DataFrame(
      {
          "Left Hip": left_hip_list_x,
          "Right Hip": right_hip_list_x
      }
  )

  fig_hip = px.line(motion_hip, x=motion_hip.index/fs, y=["Left Hip", "Right Hip"],
                    labels={"index": "Time (sec)"},
                    title="Motion of Hips",
                    width=800, height=400)

  # fig_hip.update_layout(font=dict(size=24))
  fig_hip.update_layout(
      xaxis_title="Time (sec)",
      yaxis_title="Distance",
      yaxis_title_font_size = 38, 
      xaxis_title_font_size = 38, 
      hoverlabel_font_size=38,
      title_font=dict(
          family="Courier New, monospace",
          size=40,
          color="white"
          ),
          xaxis=dict(
          tickfont=dict(
              size=28 
          ) 
          ),
          yaxis=dict(
          tickfont=dict(
          size=28 
          )
      ),

      legend=dict(
          title=dict(text='Joint', font=dict(size=36)),  # Set legend title fontsize
          font=dict(size=32)  # Set legend label fontsize
      ))
  st.plotly_chart(fig_hip, use_container_width=True)

  # Motion Knee Line Chart
  motion_knee = pd.DataFrame(
      {
          "Left Knee": left_knee_list_x,
          "Right Knee": right_knee_list_x
      }
  )

  fig_knee = px.line(motion_knee, x=motion_knee.index/fs, y=["Left Knee", "Right Knee"],
                      labels={"index": "Time (sec)"},
                      title="Motion of Knees",
                      width=800, height=400)

  fig_knee.update_layout(
      xaxis_title= "Time (sec)",
      yaxis_title="Distance",
      yaxis_title_font_size = 38, 
      xaxis_title_font_size = 38, 
      hoverlabel_font_size=38,
      title_font=dict(
          family="Courier New, monospace",
          size=40,
          color="white"
          ),
          xaxis=dict(
          tickfont=dict(
              size=28 
          ) 
          ),
          yaxis=dict(
          tickfont=dict(
          size=28 
          )
      ),

      legend=dict(
          title=dict(text='Joint', font=dict(size=36)),  # Set legend title fontsize
          font=dict(size=32)  # Set legend label fontsize
  )
  )
  st.plotly_chart(fig_knee, use_container_width=True)

  vert_oscillation = 100 * (np.max(nose_list_x) - np.min(nose_list_x)) # percent change of the video camera screen

  hip_corr = 100 * np.corrcoef(left_hip_list_x, right_hip_list_x)
  knee_corr = 100 * np.corrcoef(left_knee_list_x, right_knee_list_x)

  st.write('##### Hip and Knee Correlation')
  st.write('Hip and knee correlation is the relationship between the left and right hip and knee joints.')
  # round to 2 digits
  hip_corr = np.round(hip_corr, 2)
  knee_corr = np.round(knee_corr, 2)
  st.write(f'Hip Correlation: {hip_corr[0][1]}')
  st.write(f'Knee Correlation: {knee_corr[0][1]}')

  # DIAL PLOTS  
  dial1, dial2, dial3 = st.columns(3)
  title_font_size = 26
  with dial1:
    value = knee_corr  # Value to be displayed on the dial (e.g., gas mileage)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color="white"),
            borderwidth=2,
            bordercolor="gray",
            steps=[
                dict(range=[0, 25], color="red"),
                dict(range=[25, 50], color="orange"),
                dict(range=[50, 75], color="yellow"),
                dict(range=[75, 100], color="green")
            ],
            threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=value)
        )
    ))
    fig.update_layout(
        title={'text': "Knee Symmetry Score", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        title_font_size = title_font_size,      
        font=dict(size=24)
    )
    st.plotly_chart(fig, use_container_width=True)
    # if hip drive is low, recommend hip mobility exercises & strengthening, if really low, also recommend arm swing exercises
    # recommended drills: SuperMarios, Hill Sprints, single leg hops, deadlifts
    if knee_corr < 60:
        st.write("## <div style='text-align: center;'><span style='color: red;'>POOR</span>", unsafe_allow_html=True)
    elif knee_corr > 60 and knee_corr < 80:
        st.write("## <div style='text-align: center;'><span style='color: yellow;'>AVERAGE</span>", unsafe_allow_html=True)
    elif knee_corr > 80:
      st.write("## <div style='text-align: center;'><span style='color: green;'>GOOD</span>", unsafe_allow_html=True)

    with st.expander('Knee Symmetry Score'):
        st.write('Knee Mobility is the ability of the knee joint to move through its full range of motion. Knee mobility is important for running because it allows you to generate power from your knees and quads. A lack of knee mobility can lead to overstriding, which can lead to knee pain and shin splints. Knee mobility exercises can help improve your running form and prevent injuries.')
        st.write('Recommended Drills')
        st.write('* Depth Squat')

  with dial2:
    value = hip_corr
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color="white"),
            borderwidth=2,
            bordercolor="gray",
            steps=[
                dict(range=[0, 25], color="red"),
                dict(range=[25, 50], color="orange"),
                dict(range=[50, 75], color="yellow"),
                dict(range=[75, 100], color="green")
            ],
            threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=value)
        )
    ))
    fig.update_layout(
        title={'text': "Foot Strike Score", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        title_font_size = title_font_size,
        font=dict(size=24)
    )
    st.plotly_chart(fig, use_container_width=True)
    if hip_corr < 60:
        st.write("## <div style='text-align: center;'><span style='color: red;'>POOR</span>", unsafe_allow_html=True)
    elif hip_corr > 60 and hip_corr < 80:
        st.write("## <div style='text-align: center;'><span style='color: yellow;'>AVERAGE</span>", unsafe_allow_html=True)
    elif hip_corr > 80:
        st.write("## <div style='text-align: center;'><span style='color: green;'>GOOD</span>", unsafe_allow_html=True)

    with st.expander("Hip Symmetry Score"):
        # st.plotly_chart(fig, use_container_width=True)
        st.write('Hip Mobility is the ability of the hip joint to move through its full range of motion. Hip mobility is important for running because it allows you to generate power from your hips and glutes. A lack of hip mobility can lead to overstriding, which can lead to knee pain and shin splints. Hip mobility exercises can help improve your running form and prevent injuries.')
        # recommended exercises for the hip
        st.write('##### Recommended Drills')
        st.write('* Bird Dogs')
        st.write('* Hip Circles')
        st.write('* Hip Flexor Stretch')
        st.write('* Hip Hinge')

  # radar plot for vert_oscillation
  with dial3:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=vert_oscillation,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color="white"),
            borderwidth=2,
            bordercolor="gray",
            steps=[
                dict(range=[0, 25], color="red"),
                dict(range=[25, 50], color="orange"),
                dict(range=[50, 75], color="yellow"),
                dict(range=[75, 100], color="green")
            ],
            threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=value3)
        )
    ))
    fig.update_layout(
        title={'text': "VERTICAL OSCICILLATION", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        title_font_size = title_font_size,
        font=dict(size=28)
    )
    st.plotly_chart(fig, use_container_width=True)
    # if arm swing is low, then hip drive is low. Recommend hip mobility exercises and arm swing exercises
    if vert_oscillation < 10:
        st.write("## <div style='text-align: center;'><span style='color: green;'>GOOD</span>", unsafe_allow_html=True)
    elif vert_oscillation > 10 and vert_oscillation < 25:
        st.write("## <div style='text-align: center;'><span style='color: yellow;'>AVERAGE</span>", unsafe_allow_html=True)
    elif vert_oscillation > 25:
        st.write("## <div style='text-align: center;'><span style='color: red;'>BAD</span>", unsafe_allow_html=True)

    with st.expander("Vertical Oscillation"):
        st.plotly_chart(fig, use_container_width=True)
        st.write('Vertical Oscillation is the vertical movement of the body center of mass. It is the distance between the highest and lowest points of the body center of mass during running.')



# PLOT 3 DIAL PLOTS BASED ON ARMSWING, HIP DRIVE, AND FOOTSTRIKE SCORES!!
# SLOW pace arcs of motion: ankle, 50 degrees; knee, 95 degrees; and hip, 40 degrees.
# FAST pace, the hip required more extension in early swing; the hip and knee required more flexion in middle and late swings. The fact that ankle motion did not change with the different speeds gave credence to the belief that push-off, or toe-off, is not the source of power in running
#   Example data
#   left_knee_norm, right_knee_norm, left_hip_norm, right_hip_norm = 0.8, 0.7, 0.9, 0.75
#   left_hip_list_x = [1, 2, 3, 4, 5]
#   right_hip_list_x = [1, 2, 3, 4, 5]
#   left_knee_list_x = [1, 2, 3, 4, 5]
#   right_knee_list_x = [1, 2, 3, 4, 5]

# Call the function to plot the results
# plot_results(left_knee_norm, right_knee_norm, left_hip_norm, right_hip_norm,
# left_hip_list_x, right_hip_list_x, left_knee_list_x, right_knee_list_x)
# ======== END MOVENET ========

#       st.write('##### Recommended Drills')
#       st.write('* Arm Swings')
#       st.write('* SuperMarios')
#       st.write('* Hill Sprints')
#       st.write('* Single Leg Hops')
#       st.write('* Deadlifts')
#       st.write('##### Recommended Exercises')
#       st.write('* Banded Hip Thrusts')
#       st.write('* Banded Lateral Walks')
#       st.write('* Banded Monster Walks')
#       st.write('* Banded Squats')
#       st.write('* Banded Glute Bridges')
#       st.write('* Banded Clamshells')
#       st.write('* Banded Fire Hydrants')
#       st.write('* Banded Kickbacks')
#       st.write('* Banded Donkey Kicks')
#       st.write('* Banded Side Leg Raises')
#       st.write('* Banded Leg Extensions')
#       st.write('* Banded Leg Curls')
#       st.write('* Banded Hip Abductions')
#       st.write('* Banded Hip Adductions')
#       st.write('* Banded Hip Flexions')
#       st.write('* Banded Hip Extensions')
#       st.write('* Banded Hip Rotations')
     
# # ========== DIGITAL ATHLETE ==========
