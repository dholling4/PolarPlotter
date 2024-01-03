# import contextlib
import pandas as pd
import streamlit as st
# import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

VERSION = "0.3.1"

# THE DIGITAL ATHLETE ========================================
st.markdown("""# Baseline 🧘‍♀️🤸‍♂️📊""")
# url = "https://www.acsm.org/docs/default-source/regional-chapter-individual-folders/northland/nacsm--wes-e--fms9a9b0c1f5032400f990d8b57689b0158.pdf?sfvrsn=3668bbe0_0"
# st.markdown(" ### Check out the [Functional Movement Screen](%s)" % url)
st.sidebar.markdown("# Baseline 🧘‍♀️🤸‍♂️📊")
squat_url2 = "https://drive.google.com/uc?export=download&id=1OSnkMoCvt9wfBlMyQ1XU-yTPx_TJZ52C"
squat_url = "https://drive.google.com/uc?export=download&id=1u8ikCH-r2rQZxwOlmuiietAOckrhIoA-"
gif_url = "https://drive.google.com/uc?export=download&id=16tmT7B0cVawZ2IjMkFDY7WN9Db2J1xFj"
# gif_url = "https://drive.google.com/uc?export=download&id=15rAodbylN0pB0DmY2-ao1ndYZZxib5Ki"
instructions = st.expander("Instructions")
with instructions:
  st.write("1. Record or upload your activity")
  st.write("2. Wait for the video to process")
  st.write("3. View your score and personalized results")

baseline_assessments = st.expander("Baseline Assessments")
cols1, cols2 = st.columns(2)
with baseline_assessments:
  with cols1:
    st.write("#### Depth Squat")
    st.image(squat_url, caption="Depth Squat", use_column_width=True)
  with cols2:
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

# Custom color scheme
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
  
  st.bar_chart(chart_data, x=" ", y="Stability Score", use_container_width=True, width=315)
  
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

run_front = "https://drive.google.com/uc?export=download&id=1m_ZXv8t2Fpsww7DF5Z1jt0pHsPDgRl4f"
run_side = "https://drive.google.com/uc?export=download&id=14qby7dOKonrQk68L5mATNoiEIO8zZWm8"

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


  left_knee = [4.9,11,11.1,18,20,24.14251189,23.2,19,17,16.9,19,22,24,27,28.5,38.9,50.47197318,60.06673423,63.34481952,62.82092461,59.14867077,52.89186871,46.05129804,37.88352694,30.37935605,20.94425821,15.31114056,10.97528651,9.777812427,18.1950573]
  right_knee = [3.5,12.3,12.5,19,22,25,25,20.1,17,16.8,18.8,21.1,23.8,27,27.5,38.9,51.5,61.1,66.9,67,62,53,48,37,33,23,14,9,8,12]  
  
  ## Display the chart
  st.plotly_chart(fig, use_container_width=True)

  chart_data = pd.DataFrame(
    {"Step Count": list(range(len(left_knee))), "Left Knee": left_knee, "Right Knee": right_knee}
  )
  
st.write('##### Joint Angles')
chart_data2 = pd.DataFrame(
  {"Left Knee": left_knee, "Right Knee": right_knee})

chart_type = st.selectbox('Choose a chart type', ['Line', 'Bar']) 
## Create the chart
if chart_type == 'Line':

  st.line_chart(chart_data2, y=["Left Knee", "Right Knee"],
                ) 
elif chart_type == 'Bar':
  st.bar_chart(chart_data2, y=["Left Knee", "Right Knee"])


  
dial1, dial2, dial3 = st.columns(3)
title_font_size = 24
with dial1:
  value = 75  # Value to be displayed on the dial (e.g., gas mileage)
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
  with st.expander("Arm Swing"):
      # st.plotly_chart(fig, use_container_width=True)
      st.write('Arm Swing is important during running because it helps counterbalance the motion of the legs. Arm swing should not cross the midline of the body, but have more of a forward and back rocking motion. Arm swing helps your opposite leg drive forward during toe-off. A strong the arm-swing helps power your hips and knees to drive forward during running. A weak arm swing can lead to a weak hip drive and overstriding.')
      url = "https://journals.biologists.com/jeb/article/217/14/2456/12120/The-metabolic-cost-of-human-running-is-swinging"
      st.link_button(":book: Read more about the importance of arm swing", url)


  # ----- UPLOAD AND RUN VIDEO FILE -----

# ---------- RUN VIDEO FILE --------------
from io import BytesIO
# url_squat = 'https://drive.google.com/uc?export=download&id=1OfosAFuI3UCs4TUqnxvrId4YqWjkPPwd'
url_squat = 'https://drive.google.com/uc?export=download&id=1Wonr2Xhj67gWwvE_7LLCXTWf_yn-yg7Q'

# display url_squat
# st.video(url_squat)

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

uploaded_file = st.file_uploader("Choose an image...",  type=None) # change type=None to upload any file type (iphones use .MOV) 

# ======== MoveNet ========

# import pandas as pd
# import numpy as np

# import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow_docs.vis import embed
# from matplotlib import pyplot as plt
# from matplotlib.collections import LineCollection
# import matplotlib.patches as patches
# import imageio
# from PIL import Image
# from IPython.display import HTML, display
# # --- IMPORT MOVENET ---

# module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
# input_size = 192
# def movenet(input_image):
#   """Runs detection on an input image.

#   Args:
#     input_image: A [1, height, width, 3] tensor represents the input image
#       pixels. Note that the height/width should already be resized and match the
#       expected input resolution of the model before passing into this function.

#   Returns:
#     A [1, 1, 17, 3] float numpy array representing the predicted keypoint
#     coordinates and scores.
#   """
#   model = module.signatures['serving_default']

#   # SavedModel format expects tensor type of int32.
#   input_image = tf.cast(input_image, dtype=tf.int32)
#   # Run model inference.
#   outputs = model(input_image)
#   # Output is a [1, 1, 17, 3] tensor.
#   keypoints_with_scores = outputs['output_0'].numpy()
#   return keypoints_with_scores
# # --------------------------------------
# #@title Helper functions for visualization

# # Dictionary that maps from joint names to keypoint indices.
# KEYPOINT_DICT = {
#     'nose': 0,
#     'left_eye': 1,
#     'right_eye': 2,
#     'left_ear': 3,
#     'right_ear': 4,
#     'left_shoulder': 5,
#     'right_shoulder': 6,
#     'left_elbow': 7,
#     'right_elbow': 8,
#     'left_wrist': 9,
#     'right_wrist': 10,
#     'left_hip': 11,
#     'right_hip': 12,
#     'left_knee': 13,
#     'right_knee': 14,
#     'left_ankle': 15,
#     'right_ankle': 16
# }

# KEYPOINT_EDGE_INDS_TO_COLOR = {
#     (0, 1): '#E87722',  # Auburn Orange
#     (0, 2): 'w',  # Navy Blue
#     (1, 3): '#E87722',
#     (2, 4): 'w',
#     (0, 5): '#E87722',
#     (0, 6): 'w',
#     (5, 7): '#E87722',
#     (7, 9): '#E87722',
#     (6, 8): 'w',
#     (8, 10): 'w',
#     (5, 6): '#FFD100',  # Yellow
#     (5, 11): '#E87722',
#     (6, 12): 'w',
#     (11, 12): '#FFD100',
#     (11, 13): '#E87722',
#     (13, 15): '#E87722',
#     (12, 14): 'w',
#     (14, 16): 'w'
# }

# def _keypoints_and_edges_for_display(keypoints_with_scores,
#                                      height,
#                                      width,
#                                      keypoint_threshold=0.11):
#   """Returns high confidence keypoints and edges for visualization.

#   Args:
#     keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
#       the keypoint coordinates and scores returned from the MoveNet model.
#     height: height of the image in pixels.
#     width: width of the image in pixels.
#     keypoint_threshold: minimum confidence score for a keypoint to be
#       visualized.

#   Returns:
#     A (keypoints_xy, edges_xy, edge_colors) containing:
#       * the coordinates of all keypoints of all detected entities;
#       * the coordinates of all skeleton edges of all detected entities;
#       * the colors in which the edges should be plotted.
#   """
#   keypoints_all = []
#   keypoint_edges_all = []
#   edge_colors = []
#   num_instances, _, _, _ = keypoints_with_scores.shape
#   for idx in range(num_instances):
#     kpts_x = keypoints_with_scores[0, idx, :, 1]
#     kpts_y = keypoints_with_scores[0, idx, :, 0]
#     kpts_scores = keypoints_with_scores[0, idx, :, 2]
#     kpts_absolute_xy = np.stack(
#         [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
#     kpts_above_thresh_absolute = kpts_absolute_xy[
#         kpts_scores > keypoint_threshold, :]
#     keypoints_all.append(kpts_above_thresh_absolute)

#     for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
#       if (kpts_scores[edge_pair[0]] > keypoint_threshold and
#           kpts_scores[edge_pair[1]] > keypoint_threshold):
#         x_start = kpts_absolute_xy[edge_pair[0], 0]
#         y_start = kpts_absolute_xy[edge_pair[0], 1]
#         x_end = kpts_absolute_xy[edge_pair[1], 0]
#         y_end = kpts_absolute_xy[edge_pair[1], 1]
#         line_seg = np.array([[x_start, y_start], [x_end, y_end]])
#         keypoint_edges_all.append(line_seg)
#         edge_colors.append(color)
#   if keypoints_all:
#     keypoints_xy = np.concatenate(keypoints_all, axis=0)
#   else:
#     keypoints_xy = np.zeros((0, 17, 2))

#   if keypoint_edges_all:
#     edges_xy = np.stack(keypoint_edges_all, axis=0)
#   else:
#     edges_xy = np.zeros((0, 2, 2))
#   return keypoints_xy, edges_xy, edge_colors


# def draw_prediction_on_image(
#     image, keypoints_with_scores, crop_region=None, close_figure=False,
#     output_image_height=None):
#   """Draws the keypoint predictions on image.

#   Args:
#     image: A numpy array with shape [height, width, channel] representing the
#       pixel values of the input image.
#     keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
#       the keypoint coordinates and scores returned from the MoveNet model.
#     crop_region: A dictionary that defines the coordinates of the bounding box
#       of the crop region in normalized coordinates (see the init_crop_region
#       function below for more detail). If provided, this function will also
#       draw the bounding box on the image.
#     output_image_height: An integer indicating the height of the output image.
#       Note that the image aspect ratio will be the same as the input image.

#   Returns:
#     A numpy array with shape [out_height, out_width, channel] representing the
#     image overlaid with keypoint predictions.
#   """
#   height, width, channel = image.shape
#   aspect_ratio = float(width) / height
#   fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
#   # To remove the huge white borders
#   fig.tight_layout(pad=0)
#   ax.margins(0)
#   ax.set_yticklabels([])
#   ax.set_xticklabels([])
#   plt.axis('off')

#   im = ax.imshow(image)
#   line_segments = LineCollection([], linewidths=(4), linestyle='solid')
#   ax.add_collection(line_segments)
#   # Turn off tick labels
#   scat = ax.scatter([], [], s=100, color='#FF1493', zorder=3)

#   (keypoint_locs, keypoint_edges,
#    edge_colors) = _keypoints_and_edges_for_display(
#        keypoints_with_scores, height, width)

#   line_segments.set_segments(keypoint_edges)
#   line_segments.set_color(edge_colors)
#   if keypoint_edges.shape[0]:
#     line_segments.set_segments(keypoint_edges)
#     line_segments.set_color(edge_colors)
#   if keypoint_locs.shape[0]:
#     scat.set_offsets(keypoint_locs)

#   if crop_region is not None:
#     xmin = max(crop_region['x_min'] * width, 0.0)
#     ymin = max(crop_region['y_min'] * height, 0.0)
#     rec_width = min(crop_region['x_max'], 0.99) * width - xmin
#     rec_height = min(crop_region['y_max'], 0.99) * height - ymin
#     rect = patches.Rectangle(
#         (xmin,ymin),rec_width,rec_height,
#         linewidth=1,edgecolor='b',facecolor='none')
#     ax.add_patch(rect)

#   fig.canvas.draw()
#   image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#   image_from_plot = image_from_plot.reshape(
#       fig.canvas.get_width_height()[::-1] + (3,))
#   plt.close(fig)
#   if output_image_height is not None:
#     output_image_width = int(output_image_height / height * width)
#     # image_from_plot = cv2.resize(
#     #     image_from_plot, dsize=(output_image_width, output_image_height),
#     #      interpolation=cv2.INTER_CUBIC)
#     image_from_plot = Image.fromarray(image_from_plot)
#     image_from_plot = image_from_plot.resize(
#     (output_image_width, output_image_height), resample=Image.Resampling.LANCZOS)
#     image_from_plot = np.array(image_from_plot)
    
#   return image_from_plot

# def to_gif(images, duration):
#   """Converts image sequence (4D numpy array) to gif."""
#   imageio.mimsave('./animation.gif', images, duration=duration)
#   return embed.embed_file('./animation.gif')

# def progress(value, max=100):
#   return HTML("""
#       <progress
#           value='{value}'
#           max='{max}',
#           style='width: 100%'
#       >
#           {value}
#       </progress>
#   """.format(value=value, max=max))

# # ----
# #@title Cropping Algorithm

# # Confidence score to determine whether a keypoint prediction is reliable.
# MIN_CROP_KEYPOINT_SCORE = 0.25

# def init_crop_region(image_height, image_width):
#   """Defines the default crop region.

#   The function provides the initial crop region (pads the full image from both
#   sides to make it a square image) when the algorithm cannot reliably determine
#   the crop region from the previous frame.
#   """
#   if image_width > image_height:
#     box_height = image_width / image_height
#     box_width = 1.0
#     y_min = (image_height / 2 - image_width / 2) / image_height
#     x_min = 0.0
#   else:
#     box_height = 1.0
#     box_width = image_height / image_width
#     y_min = 0.0
#     x_min = (image_width / 2 - image_height / 2) / image_width

#   return {
#     'y_min': y_min,
#     'x_min': x_min,
#     'y_max': y_min + box_height,
#     'x_max': x_min + box_width,
#     'height': box_height,
#     'width': box_width
#   }

# def torso_visible(keypoints):
#   """Checks whether there are enough torso keypoints.

#   This function checks whether the model is confident at predicting one of the
#   shoulders/hips which is required to determine a good crop region.
#   """
#   return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] >
#            MIN_CROP_KEYPOINT_SCORE or
#           keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] >
#            MIN_CROP_KEYPOINT_SCORE) and
#           (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] >
#            MIN_CROP_KEYPOINT_SCORE or
#           keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] >
#            MIN_CROP_KEYPOINT_SCORE))

# def determine_torso_and_body_range(
#     keypoints, target_keypoints, center_y, center_x):
#   """Calculates the maximum distance from each keypoints to the center location.

#   The function returns the maximum distances from the two sets of keypoints:
#   full 17 keypoints and 4 torso keypoints. The returned information will be
#   used to determine the crop size. See determineCropRegion for more detail.
#   """
#   torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
#   max_torso_yrange = 0.0
#   max_torso_xrange = 0.0
#   for joint in torso_joints:
#     dist_y = abs(center_y - target_keypoints[joint][0])
#     dist_x = abs(center_x - target_keypoints[joint][1])
#     if dist_y > max_torso_yrange:
#       max_torso_yrange = dist_y
#     if dist_x > max_torso_xrange:
#       max_torso_xrange = dist_x

#   max_body_yrange = 0.0
#   max_body_xrange = 0.0
#   for joint in KEYPOINT_DICT.keys():
#     if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
#       continue
#     dist_y = abs(center_y - target_keypoints[joint][0]);
#     dist_x = abs(center_x - target_keypoints[joint][1]);
#     if dist_y > max_body_yrange:
#       max_body_yrange = dist_y

#     if dist_x > max_body_xrange:
#       max_body_xrange = dist_x

#   return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

# def determine_crop_region(
#       keypoints, image_height,
#       image_width):
#   """Determines the region to crop the image for the model to run inference on.

#   The algorithm uses the detected joints from the previous frame to estimate
#   the square region that encloses the full body of the target person and
#   centers at the midpoint of two hip joints. The crop size is determined by
#   the distances between each joints and the center point.
#   When the model is not confident with the four torso joint predictions, the
#   function returns a default crop which is the full image padded to square.
#   """
#   target_keypoints = {}
#   for joint in KEYPOINT_DICT.keys():
#     target_keypoints[joint] = [
#       keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
#       keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
#     ]

#   if torso_visible(keypoints):
#     center_y = (target_keypoints['left_hip'][0] +
#                 target_keypoints['right_hip'][0]) / 2;
#     center_x = (target_keypoints['left_hip'][1] +
#                 target_keypoints['right_hip'][1]) / 2;

#     (max_torso_yrange, max_torso_xrange,
#       max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
#           keypoints, target_keypoints, center_y, center_x)

#     crop_length_half = np.amax(
#         [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
#           max_body_yrange * 1.2, max_body_xrange * 1.2])

#     tmp = np.array(
#         [center_x, image_width - center_x, center_y, image_height - center_y])
#     crop_length_half = np.amin(
#         [crop_length_half, np.amax(tmp)]);

#     crop_corner = [center_y - crop_length_half, center_x - crop_length_half];

#     if crop_length_half > max(image_width, image_height) / 2:
#       return init_crop_region(image_height, image_width)
#     else:
#       crop_length = crop_length_half * 2;
#       return {
#         'y_min': crop_corner[0] / image_height,
#         'x_min': crop_corner[1] / image_width,
#         'y_max': (crop_corner[0] + crop_length) / image_height,
#         'x_max': (crop_corner[1] + crop_length) / image_width,
#         'height': (crop_corner[0] + crop_length) / image_height -
#             crop_corner[0] / image_height,
#         'width': (crop_corner[1] + crop_length) / image_width -
#             crop_corner[1] / image_width
#       }
#   else:
#     return init_crop_region(image_height, image_width)

# def crop_and_resize(image, crop_region, crop_size):
#   """Crops and resize the image to prepare for the model input."""
#   boxes=[[crop_region['y_min'], crop_region['x_min'],
#           crop_region['y_max'], crop_region['x_max']]]
#   output_image = tf.image.crop_and_resize(
#       image, box_indices=[0], boxes=boxes, crop_size=crop_size)
#   return output_image

# def run_inference(movenet, image, crop_region, crop_size):
#   """Runs model inferece on the cropped region.

#   The function runs the model inference on the cropped region and updates the
#   model output to the original image coordinate system.
#   """
#   image_height, image_width, _ = image.shape
#   input_image = crop_and_resize(
#     tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
#   # Run model inference.
#   keypoints_with_scores = movenet(input_image)
#   # Update the coordinates.
#   for idx in range(17):
#     keypoints_with_scores[0, 0, idx, 0] = (
#         crop_region['y_min'] * image_height +
#         crop_region['height'] * image_height *
#         keypoints_with_scores[0, 0, idx, 0]) / image_height
#     keypoints_with_scores[0, 0, idx, 1] = (
#         crop_region['x_min'] * image_width +
#         crop_region['width'] * image_width *
#         keypoints_with_scores[0, 0, idx, 1]) / image_width
#   return keypoints_with_scores


# if uploaded_file is not None:
  # update for .MOV  ========= START =========
  # import moviepy
  # from moviepy.editor import VideoFileClip
  # path = "https://drive.google.com/uc?export=download&id=1UOtno-A__uflgVLECYsEOUWFabeUZX45"
  # file_name = uploaded_file.name
  # st.write(uploaded_file.name)
  # save_directory = "./"
  # file_path = save_directory + file_name
  # with open(file_path, "wb") as f:
  #     f.write(uploaded_file.read())
  # st.success(f"File saved to: {file_path}")
  # path2mov = r"/workspaces/PolarPlotter/" + str(file_name) 
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

  # image_content = uploaded_file.read()
  # image = tf.image.decode_gif(image_content)
  # num_frames, image_height, image_width, _ = image.shape
  # crop_region = init_crop_region(image_height, image_width)

  # nose_list_x, left_shoulder_list_x, right_shoulder_list_x,left_elbow_list_x, right_elbow_list_x, left_wrist_list_x, right_wrist_list_x = [],[],[],[],[],[],[]
  # left_ankle_list_x, right_ankle_list_x, left_hip_list_x, right_hip_list_x, left_knee_list_x, right_knee_list_x = [], [], [], [], [] ,[]

  # nose_list_y, left_shoulder_list_y, right_shoulder_list_y,left_elbow_list_y, right_elbow_list_y, left_wrist_list_y, right_wrist_list_y = [],[],[],[],[],[],[]
  # left_ankle_list_y, right_ankle_list_y, left_hip_list_y, right_hip_list_y, left_knee_list_y, right_knee_list_y = [], [], [], [], [] ,[]

  # nose_list_conf, left_shoulder_list_conf, right_shoulder_list_conf,left_elbow_list_conf, right_elbow_list_conf, left_wrist_list_conf, right_wrist_list_conf = [],[],[],[],[],[],[]
  # left_ankle_list_conf, right_ankle_list_conf, left_hip_list_conf, right_hip_list_conf, left_knee_list_conf, right_knee_list_conf = [], [], [], [], [] ,[]

  # left_knee_angle_list_x, right_knee_angle_list_x = [], []
  # left_knee_deg_list, right_knee_deg_list = [], []
  # output_images = []
  # bar = display(progress(0, image.shape[0]-1), display_id=True)
  # for frame_idx in range(num_frames):
  #   keypoints_with_scores = run_inference(
  #       movenet, image[frame_idx, :, :, :], crop_region,
  #       crop_size=[input_size, input_size])
  #   output_images.append(draw_prediction_on_image(
  #       image[frame_idx, :, :, :].numpy().astype(np.int32),
  #       keypoints_with_scores, crop_region=None,
  #       close_figure=True, output_image_height=300))
  #   crop_region = determine_crop_region(
  #       keypoints_with_scores, image_height, image_width)
  #   # bar.update(progress(frame_idx, num_frames-1))

  #   nose_x = keypoints_with_scores[0,0,0,0]
  #   left_shoulder_x = keypoints_with_scores[0,0,5,0]
  #   right_shoulder_x = keypoints_with_scores[0,0,6,0]
  #   left_elbow_x = keypoints_with_scores[0,0,7,0]
  #   right_elbow_x = keypoints_with_scores[0,0,8,0]
  #   left_wrist_x = keypoints_with_scores[0,0,9,0]
  #   right_wrist_x = keypoints_with_scores[0,0,10,0]
  #   # HIPS
  #   left_hip_x = keypoints_with_scores[0,0,11,0]
  #   right_hip_x = keypoints_with_scores[0,0,12,0]
  #   # ANKLES
  #   left_ankle_x = keypoints_with_scores[0,0,15,0]
  #   right_ankle_x = keypoints_with_scores[0,0,16,0]
  #   # KNEES
  #   left_knee_x = keypoints_with_scores[0,0,13,0]
  #   right_knee_x = keypoints_with_scores[0,0,14,0]

  #   # Append keypoints to list
  #   nose_list_x.append(nose_x)
  #   left_shoulder_list_x.append(left_shoulder_x)
  #   right_shoulder_list_x.append(right_shoulder_x)
  #   left_elbow_list_x.append(left_elbow_x)
  #   right_elbow_list_x.append(right_elbow_x)
  #   left_wrist_list_x.append(left_wrist_x)
  #   right_wrist_list_x.append(right_wrist_x)
  #   left_ankle_list_x.append(left_ankle_x)
  #   right_ankle_list_x.append(right_ankle_x)
  #   left_knee_list_x.append(left_knee_x)
  #   right_knee_list_x.append(right_knee_x)
  #   left_hip_list_x.append(left_hip_x)
  #   right_hip_list_x.append(right_hip_x)


  #   nose_y = keypoints_with_scores[0,0,0,1]
  #   left_shoulder_y = keypoints_with_scores[0,0,5,1]
  #   right_shoulder_y = keypoints_with_scores[0,0,6,1]
  #   left_elbow_y = keypoints_with_scores[0,0,7,1]
  #   right_elbow_y = keypoints_with_scores[0,0,8,1]
  #   left_wrist_y = keypoints_with_scores[0,0,9,1]
  #   right_wrist_y = keypoints_with_scores[0,0,10,1]
  #   # HIPS
  #   left_hip_y = keypoints_with_scores[0,0,11,1]
  #   right_hip_y = keypoints_with_scores[0,0,12,1]
  #   # ANKLES
  #   left_ankle_y = keypoints_with_scores[0,0,15,1]
  #   right_ankle_y = keypoints_with_scores[0,0,16,1]
  #   # KNEES
  #   left_knee_y = keypoints_with_scores[0,0,13,1]
  #   right_knee_y = keypoints_with_scores[0,0,14,1]

  #   # Append keypoints to list
  #   nose_list_y.append(nose_y)
  #   left_shoulder_list_y.append(left_shoulder_y)
  #   right_shoulder_list_y.append(right_shoulder_y)
  #   left_elbow_list_y.append(left_elbow_y)
  #   right_elbow_list_y.append(right_elbow_y)
  #   left_wrist_list_y.append(left_wrist_y)
  #   right_wrist_list_y.append(right_wrist_y)
  #   left_ankle_list_y.append(left_ankle_y)
  #   right_ankle_list_y.append(right_ankle_y)
  #   left_knee_list_y.append(left_knee_y)
  #   right_knee_list_y.append(right_knee_y)
  #   left_hip_list_y.append(left_hip_y)
  #   right_hip_list_y.append(right_hip_y)

  #   nose_c = keypoints_with_scores[0,0,0,2]
  #   left_shoulder_c = keypoints_with_scores[0,0,5,2]
  #   right_shoulder_c = keypoints_with_scores[0,0,6,2]
  #   left_elbow_c = keypoints_with_scores[0,0,7,2]
  #   right_elbow_c = keypoints_with_scores[0,0,8,2]
  #   left_wrist_c = keypoints_with_scores[0,0,9,2]
  #   right_wrist_c = keypoints_with_scores[0,0,10,2]
  #   # HIPS
  #   left_hip_c = keypoints_with_scores[0,0,11,2]
  #   right_hip_c = keypoints_with_scores[0,0,12,2]
  #   # ANKLES
  #   left_ankle_c = keypoints_with_scores[0,0,15,2]
  #   right_ankle_c = keypoints_with_scores[0,0,16,2]
  #   # KNEES
  #   left_knee_c = keypoints_with_scores[0,0,13,2]
  #   right_knee_c = keypoints_with_scores[0,0,14,2]

  #   # Append keypoints to list
  #   nose_list_conf.append(nose_c)
  #   left_shoulder_list_conf.append(left_shoulder_c)
  #   right_shoulder_list_conf.append(right_shoulder_c)
  #   left_elbow_list_conf.append(left_elbow_c)
  #   right_elbow_list_conf.append(right_elbow_c)
  #   left_wrist_list_conf.append(left_wrist_c)
  #   right_wrist_list_conf.append(right_wrist_c)
  #   left_ankle_list_conf.append(left_ankle_c)
  #   right_ankle_list_conf.append(right_ankle_c)
  #   left_knee_list_conf.append(left_knee_c)
  #   right_knee_list_conf.append(right_knee_c)
  #   left_hip_list_conf.append(left_hip_c)
  #   right_hip_list_conf.append(right_hip_c)
    
  # output = np.stack(output_images, axis=0)
  # image_capture = to_gif(output, duration=100)

  # def euclidean_distance(array):
  #   euclidean_distance = np.linalg.norm(array)
  #   return euclidean_distance

  # # get the euclidean distance of the medio-lateral directions
  # left_knee_norm = euclidean_distance(left_knee_list_y)
  # right_knee_norm = euclidean_distance(right_knee_list_y)
  # left_hip_norm = euclidean_distance(left_hip_list_y)
  # right_hip_norm = euclidean_distance(right_hip_list_y)
  # left_shoulder_norm = euclidean_distance(left_shoulder_list_y)
  # right_shoulder_norm = euclidean_distance(right_shoulder_list_y) 
  # st.write(image_capture)

  # """
  # ### Video Results
  # """
  # chart_data = pd.DataFrame(
  #   {
  #       "Joint": ['Left Knee', 'Right Knee', 'Left Hip', 'Right Hip'],
  #       "Stability Score": [left_knee_norm,right_knee_norm,left_hip_norm,right_hip_norm],
  #   }
  # )

  # st.bar_chart(chart_data, x="Joint", y="Stability Score")

  # motion_hip = pd.DataFrame( 
  #     { "Left Hip": left_hip_list_x,
  #       "Right HIp" : right_hip_list_x
  #     }
  # )

  # motion_knee = pd.DataFrame( 
  #     { "Left Knee": left_knee_list_x,
  #       "Right Knee" : right_knee_list_x
  #     }
  # )
  # st.line_chart(motion_knee)
  # st.line_chart(motion_hip)



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

# # # ---------- FUNCTIONS ----------
# # def _reset() -> None:
# #     st.session_state["title"] = ""
#     st.session_state["hovertemplate"] = "%{theta}: %{r}"
#     st.session_state["opacity"] = st.session_state["marker_opacity"] = st.session_state[
#         "line_smoothing"
#     ] = 1
#     st.session_state["mode"] = ["lines", "markers"]
#     st.session_state["marker_color"] = st.session_state[
#         "line_color"
#     ] = st.session_state["fillcolor"] = "#636EFA"
#     st.session_state["marker_size"] = 6
#     st.session_state["marker_symbol"] = "circle"
#     st.session_state["line_dash"] = "solid"
#     st.session_state["line_shape"] = "linear"
#     st.session_state["line_width"] = 2
#     st.session_state["fill_opacity"] = 0.5


# # ---------- HEADER ----------
# st.title("🕸️ Welcome to Polar Plotter!")
# st.subheader("Easily create rich polar/radar/spider plots.")


# # ---------- DATA ENTRY ----------
# option = st.radio(
#     label="Enter data",
#     options=(
#         "Play with example data 💡",
#         "Upload an excel file ⬆️",
#         "Add data manually ✍️",
#     ),
#     help="Uploaded files are deleted from the server when you\n* upload another file\n* clear the file uploader\n* close the browser tab",
# )

# if option == "Upload an excel file ⬆️":
#     if uploaded_file := st.file_uploader(
#         label="Upload a file. File should have the format: Label|Value",
#         type=["xlsx", "csv", "xls"],
#     ):
#         input_df = pd.read_excel(uploaded_file)
#         st.dataframe(input_df, hide_index=True)

# else:
#     if option == "Add data manually ✍️":
#         _df = pd.DataFrame(columns=["Label", "Value"]).reset_index(drop=True)

#     else:
#         _df = pd.DataFrame(
#             {
#                 "Skill": [
#                     "Computer Vision",
#                     "Prototyping",
#                     "Classic ML",
#                     "AE/VAE",
#                     "Visualization",
#                     "Storytelling",
#                     "BI",
#                     "SQL",
#                     "Deploy",
#                     "MLOps",
#                     "Excel",
#                     "Reporting",
#                     "ViT",
#                     "Diffusers",
#                     "Python",
#                     "NLP",
#                 ],
#                 "Proficiency": [
#                     4.2,
#                     4.7,
#                     2.3,
#                     4,
#                     1.9,
#                     2.4,
#                     0.5,
#                     0.6,
#                     0.2,
#                     0.3,
#                     5,
#                     1.6,
#                     4,
#                     2.4,
#                     3.4,
#                     3,
#                 ],
#             }
#         )

#     input_df = st.data_editor(
#         _df,
#         num_rows="dynamic",
#         hide_index=True,
#     )

# # ---------- SIDEBAR ----------
# with open("sidebar.html", "r", encoding="UTF-8") as sidebar_file:
#     sidebar_html = sidebar_file.read().replace("{VERSION}", VERSION)

# ## ---------- Customization options ----------
# with st.sidebar:
#     with st.expander("Customization options"):
#         title = st.text_input(
#             label="Plot title",
#             value="Job Requirements" if option == "Play with example data 💡" else "",
#             help="Sets the plot title.",
#             key="title",
#         )

#         opacity = st.slider(
#             label="Opacity",
#             min_value=0.0,
#             max_value=1.0,
#             value=1.0,
#             step=0.1,
#             help="Sets the opacity of the trace",
#             key="opacity",
#         )

#         mode = st.multiselect(
#             label="Mode",
#             options=["lines", "markers"],
#             default=["lines", "markers"],
#             help='Determines the drawing mode for this scatter trace. If the provided `mode` includes "text" then the `text` elements appear at the coordinates. '
#             'Otherwise, the `text` elements appear on hover. If there are less than 20 points and the trace is not stacked then the default is "lines+markers". Otherwise, "lines".',
#             key="mode",
#         )

#         hovertemplate = st.text_input(
#             label="Hover template",
#             value="%{theta}: %{r}",
#             help=r"""Template string used for rendering the information that appear on hover box. Note that this will override `hoverinfo`.
#             Variables are inserted using %{variable}, for example "y: %{y}" as well as %{xother}, {%_xother}, {%_xother_}, {%xother_}.
#             When showing info for several points, "xother" will be added to those with different x positions from the first point.
#             An underscore before or after "(x|y)other" will add a space on that side, only when this field is shown.
#             Numbers are formatted using d3-format's syntax %{variable:d3-format}, for example "Price: %{y:$.2f}".
#             https://github.com/d3/d3-format/tree/v1.4.5#d3-format for details on the formatting syntax.
#             Dates are formatted using d3-time-format's syntax %{variable|d3-time-format}, for example "Day: %{2019-01-01|%A}".
#             https://github.com/d3/d3-time-format/tree/v2.2.3#locale_format for details on the date formatting syntax.
#             The variables available in `hovertemplate` are the ones emitted as event data described at this link https://plotly.com/javascript/plotlyjs-events/#event-data.
#             Additionally, every attributes that can be specified per-point (the ones that are `arrayOk: True`) are available.
#             Anything contained in tag `<extra>` is displayed in the secondary box, for example "<extra>{fullData.name}</extra>".
#             To hide the secondary box completely, use an empty tag `<extra></extra>`.""",
#             key="hovertemplate",
#         )

#         marker_color = st.color_picker(
#             label="Marker color",
#             value="#636EFA",
#             key="marker_color",
#             help="Sets the marker color",
#         )

#         marker_opacity = st.slider(
#             label="Marker opacity",
#             min_value=0.0,
#             max_value=1.0,
#             value=1.0,
#             step=0.1,
#             help="Sets the marker opacity",
#             key="marker_opacity",
#         )

#         marker_size = st.slider(
#             label="Marker size",
#             min_value=0,
#             max_value=10,
#             value=6,
#             step=1,
#             help="Sets the marker size (in px)",
#             key="marker_size",
#         )

#         marker_symbol = st.selectbox(
#             label="Marker symbol",
#             index=24,
#             options=[
#                 "arrow",
#                 "arrow-bar-down",
#                 "arrow-bar-down-open",
#                 "arrow-bar-left",
#                 "arrow-bar-left-open",
#                 "arrow-bar-right",
#                 "arrow-bar-right-open",
#                 "arrow-bar-up",
#                 "arrow-bar-up-open",
#                 "arrow-down",
#                 "arrow-down-open",
#                 "arrow-left",
#                 "arrow-left-open",
#                 "arrow-open",
#                 "arrow-right",
#                 "arrow-right-open",
#                 "arrow-up",
#                 "arrow-up-open",
#                 "arrow-wide",
#                 "arrow-wide-open",
#                 "asterisk",
#                 "asterisk-open",
#                 "bowtie",
#                 "bowtie-open",
#                 "circle",
#                 "circle-cross",
#                 "circle-cross-open",
#                 "circle-dot",
#                 "circle-open",
#                 "circle-open-dot",
#                 "circle-x",
#                 "circle-x-open",
#                 "cross",
#                 "cross-dot",
#                 "cross-open",
#                 "cross-open-dot",
#                 "cross-thin",
#                 "cross-thin-open",
#                 "diamond",
#                 "diamond-cross",
#                 "diamond-cross-open",
#                 "diamond-dot",
#                 "diamond-open",
#                 "diamond-open-dot",
#                 "diamond-tall",
#                 "diamond-tall-dot",
#                 "diamond-tall-open",
#                 "diamond-tall-open-dot",
#                 "diamond-wide",
#                 "diamond-wide-dot",
#                 "diamond-wide-open",
#                 "diamond-wide-open-dot",
#                 "diamond-x",
#                 "diamond-x-open",
#                 "hash",
#                 "hash-dot",
#                 "hash-open",
#                 "hash-open-dot",
#                 "hexagon",
#                 "hexagon2",
#                 "hexagon2-dot",
#                 "hexagon2-open",
#                 "hexagon2-open-dot",
#                 "hexagon-dot",
#                 "hexagon-open",
#                 "hexagon-open-dot",
#                 "hexagram",
#                 "hexagram-dot",
#                 "hexagram-open",
#                 "hexagram-open-dot",
#                 "hourglass",
#                 "hourglass-open",
#                 "line-ew",
#                 "line-ew-open",
#                 "line-ne",
#                 "line-ne-open",
#                 "line-ns",
#                 "line-ns-open",
#                 "line-nw",
#                 "line-nw-open",
#                 "octagon",
#                 "octagon-dot",
#                 "octagon-open",
#                 "octagon-open-dot",
#                 "pentagon",
#                 "pentagon-dot",
#                 "pentagon-open",
#                 "pentagon-open-dot",
#                 "square",
#                 "square-cross",
#                 "square-cross-open",
#                 "square-dot",
#                 "square-open",
#                 "square-open-dot",
#                 "square-x",
#                 "square-x-open",
#                 "star",
#                 "star-diamond",
#                 "star-diamond-dot",
#                 "star-diamond-open",
#                 "star-diamond-open-dot",
#                 "star-dot",
#                 "star-open",
#                 "star-open-dot",
#                 "star-square",
#                 "star-square-dot",
#                 "star-square-open",
#                 "star-square-open-dot",
#                 "star-triangle-down",
#                 "star-triangle-down-dot",
#                 "star-triangle-down-open",
#                 "star-triangle-down-open-dot",
#                 "star-triangle-up",
#                 "star-triangle-up-dot",
#                 "star-triangle-up-open",
#                 "star-triangle-up-open-dot",
#                 "triangle-down",
#                 "triangle-down-dot",
#                 "triangle-down-open",
#                 "triangle-down-open-dot",
#                 "triangle-left",
#                 "triangle-left-dot",
#                 "triangle-left-open",
#                 "triangle-left-open-dot",
#                 "triangle-ne",
#                 "triangle-ne-dot",
#                 "triangle-ne-open",
#                 "triangle-ne-open-dot",
#                 "triangle-nw",
#                 "triangle-nw-dot",
#                 "triangle-nw-open",
#                 "triangle-nw-open-dot",
#                 "triangle-right",
#                 "triangle-right-dot",
#                 "triangle-right-open",
#                 "triangle-right-open-dot",
#                 "triangle-se",
#                 "triangle-se-dot",
#                 "triangle-se-open",
#                 "triangle-se-open-dot",
#                 "triangle-sw",
#                 "triangle-sw-dot",
#                 "triangle-sw-open",
#                 "triangle-sw-open-dot",
#                 "triangle-up",
#                 "triangle-up-dot",
#                 "triangle-up-open",
#                 "triangle-up-open-dot",
#                 "x",
#                 "x-dot",
#                 "x-open",
#                 "x-open-dot",
#                 "x-thin",
#                 "x-thin-open",
#                 "y-down",
#                 "y-down-open",
#                 "y-left",
#                 "y-left-open",
#                 "y-right",
#                 "y-right-open",
#                 "y-up",
#                 "y-up-open",
#             ],
#             help="""Sets the marker symbol type. Adding 100 is equivalent to appending "-open" to a symbol name.
#             Adding 200 is equivalent to appending "-dot" to a symbol name. Adding 300 is equivalent to appending "-open-dot" or "dot-open" to a symbol name.""",
#             key="marker_symbol",
#         )

#         line_color = st.color_picker(
#             label="Line color",
#             value="#636EFA",
#             key="line_color",
#             help="Sets the line color",
#         )

#         line_dash = st.selectbox(
#             label="Line dash",
#             options=["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"],
#             help="""Sets the dash style of lines.
#             Set to a dash type string ("solid", "dot", "dash", "longdash", "dashdot", or "longdashdot") or a dash length list in px (eg "5px,10px,2px,2px").""",
#             key="line_dash",
#         )

#         line_shape = st.selectbox(
#             label="Line shape",
#             options=["linear", "spline"],
#             help="""Determines the line shape. With "spline" the lines are drawn using spline interpolation.
#             The other available values correspond to step-wise line shapes.""",
#             key="line_shape",
#         )

#         line_smoothing = st.slider(
#             label="Line smoothing",
#             min_value=0.0,
#             max_value=1.3,
#             value=1.0,
#             step=0.1,
#             help="""Has an effect only if `shape` is set to "spline" Sets the amount of smoothing.
#             "0" corresponds to no smoothing (equivalent to a "linear" shape).""",
#             key="line_smoothing",
#             disabled=line_shape == "linear",
#         )

#         line_width = st.slider(
#             label="Line width",
#             min_value=0,
#             max_value=10,
#             value=2,
#             step=1,
#             help="""Sets the line width (in px).""",
#             key="line_width",
#         )

#         fillcolor = st.color_picker(
#             label="Fill color",
#             value="#636EFA",
#             key="fillcolor",
#             help="Sets the fill color. Defaults to a half-transparent variant of the line color, marker color, or marker line color, whichever is available.",
#         )

#         fill_opacity = st.slider(
#             label="Fill opacity",
#             min_value=0.0,
#             max_value=1.0,
#             value=0.5,
#             step=0.1,
#             help="""Sets the fill opacity.""",
#             key="fill_opacity",
#         )

#         rgba = tuple(
#             (
#                 [int(fillcolor.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
#                 + [fill_opacity]
#             )
#         )

#         st.button(
#             "↩️ Reset to defaults",
#             on_click=_reset,
#             use_container_width=True,
#         )

#     st.components.v1.html(sidebar_html, height=750)

# # ---------- VISUALIZATION ----------
# with contextlib.suppress(IndexError, NameError):
#     labels = list(input_df[input_df.columns[0]])
#     # To close the polygon
#     values = list(input_df[input_df.columns[1]])
#     labels = (labels + [labels[0]])[::-1]
#     values = (values + [values[0]])[::-1]

#     data = go.Scatterpolar(
#         r=values,
#         theta=labels,
#         mode="none" if mode == [] else "+".join(mode),
#         opacity=opacity,
#         hovertemplate=hovertemplate + "<extra></extra>",
#         marker_color=marker_color,
#         marker_opacity=marker_opacity,
#         marker_size=marker_size,
#         marker_symbol=marker_symbol,
#         line_color=line_color,
#         line_dash=line_dash,
#         line_shape=line_shape,
#         line_smoothing=line_smoothing,
#         line_width=line_width,
#         fill="toself",
#         fillcolor=f"RGBA{rgba}" if rgba else "RGBA(99, 110, 250, 0.5)",
#     )

#     layout = go.Layout(
#         title=dict(
#             text="Job Requirements" if option == "Play with example data 💡" else title,
#             x=0.5,
#             xanchor="center",
#         ),
#         paper_bgcolor="rgba(100,100,100,0)",
#         plot_bgcolor="rgba(100,100,100,0)",
#     )

#     fig = go.Figure(data=data, layout=layout)

#     st.plotly_chart(
#         fig,
#         use_container_width=True,
#         sharing="streamlit",
#         theme="streamlit",
#     )

#     lcol, rcol = st.columns(2)

#     with lcol:
#         with st.expander("💾Download static plot"):
#             st.write("Download static image from the plot toolbar")
#             st.image("save_as_png.png")

#     fig.write_html("interactive.html")
#     with open("interactive.html", "rb") as file:
#         rcol.download_button(
#             "💾Download interactive plot",
#             data=file,
#             mime="text/html",
#             use_container_width=True,
#         )


# POSSIBLY INCLUDE HR_NET ====================================
# pip install streamlit torch torchvision pillow

# import streamlit as st
# import torch
# from PIL import Image
# from torchvision import transforms
# from torchvision.models.detection import keypointrcnn_resnet50_fpn

# # Load HRNet model
# model = keypointrcnn_resnet50_fpn(pretrained=True)
# model.eval()

# # Define image transformation
# transform = transforms.Compose([transforms.ToTensor()])

# # Function to run HRNet on an image
# def run_hrnet(image):
#     # Preprocess the image
#     img = transform(image).unsqueeze(0)

#     # Run inference
#     with torch.no_grad():
#         prediction = model(img)

#     return prediction

# # Streamlit app
# def main():
#     st.title("HRNet Streamlit App")

#     # Upload image through Streamlit
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Read image
#         image = Image.open(uploaded_file)

#         # Display the uploaded image
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         # Run HRNet on the image
#         prediction = run_hrnet(image)

#         # Display results (modify as needed based on HRNet output)
#         st.write("HRNet results:", prediction)

# # Run the Streamlit app
# if __name__ == "__main__":
#     main()



