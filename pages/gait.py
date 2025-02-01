# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import tempfile
# import os
# from matplotlib import pyplot as plt

# # Setup MediaPipe Pose model
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# KEYPOINTS_OF_INTEREST = {
#     23: "Left Hip",
#     24: "Right Hip",
#     25: "Left Knee",
#     26: "Right Knee",
#     27: "Left Ankle",
#     28: "Right Ankle",
#     29: "Left Heel",
#     30: "Right Heel",
#     31: "Left Foot",
#     32: "Right Foot"
# }

# def calculate_angle(v1, v2):
#     """
#     Calculate the angle between two vectors using the dot product.
#     """
#     # Calculate dot product
#     dot_product = np.dot(v1, v2)
#     # Calculate magnitudes
#     magnitude_v1 = np.linalg.norm(v1)
#     magnitude_v2 = np.linalg.norm(v2)
#     # Calculate angle in radians
#     angle_radians = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
#     # Convert to degrees
#     angle_degrees = np.degrees(angle_radians)
#     return angle_degrees

# def process_video(video_path):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration = total_frames / fps  # Calculate duration in seconds
 
#     if duration > 65:
#         st.error(f"Uploaded video duration is {duration:.2f} seconds. Please upload shorter than a 60-second video.")
#     else:
#         # Calculate +- 10% of the middle seconds
#         start_time = duration*0.5 - 0.1*duration  # Middle start time in seconds 
#         end_time = duration*0.5 + 0.1*duration    # Middle end time in seconds
#         start_frame = int(start_time * fps)
#         end_frame = int(end_time * fps)
 
#     st.success(f"Upload successful!")
#     # Lists to store joint angles over time
#     left_knee_angles, right_knee_angles = [], []
#     left_hip_angles, right_hip_angles = [], []
#     left_ankle_angles, right_ankle_angles = [], []

#     # Initialize MediaPipe Pose
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         for _ in range(start_frame, end_frame):
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Convert the frame to RGB (MediaPipe expects RGB images)
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Process the frame to detect poses
#             results = pose.process(frame_rgb)
            
#             # Process detected landmarks
#             if results.pose_landmarks:
#                 # Extract landmarks
#                 landmarks = results.pose_landmarks.landmark
#                 def get_coords(landmark):
#                     return np.array([landmark.x, landmark.y])

#                 left_hip = get_coords(landmarks[23])
#                 right_hip = get_coords(landmarks[24])
#                 left_knee = get_coords(landmarks[25])
#                 right_knee = get_coords(landmarks[26])
#                 left_ankle = get_coords(landmarks[27])
#                 right_ankle = get_coords(landmarks[28])
#                 left_foot = get_coords(landmarks[31])
#                 right_foot = get_coords(landmarks[32])

#                 # Calculate vectors
#                 left_thigh_vector = left_hip - left_knee
#                 left_shank_vector = left_knee - left_ankle
#                 right_thigh_vector = right_hip - right_knee
#                 right_shank_vector = right_knee - right_ankle
#                 left_foot_vector = left_ankle - left_foot
#                 right_foot_vector = right_ankle - right_foot

#                 # Calculate joint angles
#                 left_knee_angles.append(calculate_angle(left_thigh_vector, left_shank_vector))
#                 right_knee_angles.append(calculate_angle(right_thigh_vector, right_shank_vector))
#                 left_hip_angles.append(calculate_angle(left_thigh_vector, right_thigh_vector))
#                 right_hip_angles.append(calculate_angle(right_thigh_vector, left_thigh_vector))
#                 left_ankle_angles.append(calculate_angle(left_shank_vector, left_foot_vector))
#                 right_ankle_angles.append(calculate_angle(right_shank_vector, right_foot_vector))
    
#     # Display range of motion for each joint
#     st.write("### Range of Motion (Degrees):")
#     st.write(f"Left Hip: {max(left_hip_angles) - min(left_hip_angles):.2f}")
#     st.write(f"Right Hip: {max(right_hip_angles) - min(right_hip_angles):.2f}")
#     st.write(f"Left Knee: {max(left_knee_angles) - min(left_knee_angles):.2f}")
#     st.write(f"Right Knee: {max(right_knee_angles) - min(right_knee_angles):.2f}")
#     st.write(f"Left Ankle: {max(left_ankle_angles) - min(left_ankle_angles):.2f}")
#     st.write(f"Right Ankle: {max(right_ankle_angles) - min(right_ankle_angles):.2f}")

#     # Plot joint angles over time
#     # start_time_plot = duration*0.5 - 0.1*duration
#     time = np.arange(0, len(left_hip_angles)) / 30  # Time in seconds
#     tick_fontsize=20

#     st.write('## Hip Angles')
#     fig, ax = plt.subplots(2, 1, figsize=(12, 8))
#     ax[0].plot(time, left_hip_angles, label="Left Hip")
#     ax[1].plot(time, right_hip_angles, label="Right Hip", color='orange')
#     ax[0].legend(fontsize=24)
#     ax[1].legend(fontsize=24)
#     ax[0].set_ylabel("Angle (degrees)", fontsize=20)
#     ax[1].set_xlabel("Time (s)", fontsize=20)
#     ax[1].set_ylabel("Angle (degrees)", fontsize=20)
#     ax[0].set_xlim([time[0], time[-1]])
#     ax[1].set_xlim([time[0], time[-1]])
#     ax[0].tick_params(axis='both', labelsize=tick_fontsize)
#     ax[1].tick_params(axis='both', labelsize=tick_fontsize)
#     st.pyplot(fig)

#     st.write('## Knee Angles')
#     fig2, ax = plt.subplots(2, 1, figsize=(12, 8))
#     ax[0].plot(time, left_knee_angles, label="Left Knee")
#     ax[1].plot(time, right_knee_angles, label="Right Knee", color='orange')
#     ax[0].legend(fontsize=24)
#     ax[1].legend(fontsize=24)
#     ax[0].set_ylabel("Angle (degrees)", fontsize=20)
#     ax[1].set_xlabel("Time (s)", fontsize=20)
#     ax[1].set_ylabel("Angle (degrees)", fontsize=20)
#     ax[0].set_xlim([time[0], time[-1]])
#     ax[1].set_xlim([time[0], time[-1]])
#     ax[0].tick_params(axis='both', labelsize=tick_fontsize)
#     ax[1].tick_params(axis='both', labelsize=tick_fontsize)
#     st.pyplot(fig2)

#     st.write('## Ankle Angles')
#     fig3, ax = plt.subplots(2, 1, figsize=(12, 8))
#     ax[0].plot(time, left_ankle_angles, label="Left Ankle")
#     ax[1].plot(time, right_ankle_angles, label="Right Ankle", color='orange')
#     ax[0].legend(fontsize=24)
#     ax[1].legend(fontsize=24)
#     ax[0].set_ylabel("Angle (degrees)", fontsize=20)
#     ax[1].set_xlabel("Time (s)", fontsize=20)
#     ax[1].set_ylabel("Angle (degrees)", fontsize=20)
#     ax[0].set_xlim([time[0], time[-1]])
#     ax[1].set_xlim([time[0], time[-1]])

#     ax[0].tick_params(axis='both', labelsize=tick_fontsize)
#     ax[1].tick_params(axis='both', labelsize=tick_fontsize)
#     st.pyplot(fig3)

#     # Release the video capture
#     cap.release()

def process_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    st.write(f"Total frames: {total_frames}, FPS: {fps}, Duration: {duration:.2f} seconds")

    frame_number = st.slider("Select frame", 0, total_frames - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to read the selected frame.")
        cap.release()
        return
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            annotated_frame = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_number}")
    cap.release()
    return frame_number

    # plot angles below the video



# def main():
#     st.title("Pose Estimation Gait Analysis")
    
#     # Upload video file
#     front_video = st.file_uploader("Upload side video (mp4, mov, avi, mkv file)", type=["mp4", "mov", "avi", "mkv"])
#     side_video = st.file_uploader("Upload back video (mp4, mov, avi, mkv file)", type=["mp4", "mov", "avi", "mkv"])

#     if front_video is not None:
#         # Save the uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mov") as temp_file:
#             temp_file.write(front_video.read())
#             temp_path = temp_file.name
        
#         # Process the first frame of the video
#         process_first_frame(temp_path) # display the skeleton overlay
#         st.write('### Flexion/Extension Angles')
#         process_video(temp_path) # analyze the video
        
#         # Clean up the temporary file
#         os.remove(temp_path)

#     if side_video is not None:
#         # Save the uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mov") as temp_file:
#             temp_file.write(side_video.read())
#             temp_path = temp_file.name
        
#         # Process the first frame of the video
#         process_first_frame(temp_path)
#         st.write('### Abduction/Adduction Angles')

#         process_video(temp_path)
        
#         # Clean up the temporary file
#         os.remove(temp_path)

# if __name__ == "__main__":
#     main()
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from matplotlib import pyplot as plt
import plotly.graph_objects as go

# Setup MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

KEYPOINTS_OF_INTEREST = {
    23: "Left Hip",
    24: "Right Hip",
    25: "Left Knee",
    26: "Right Knee",
    27: "Left Ankle",
    28: "Right Ankle",
    29: "Left Heel",
    30: "Right Heel",
    31: "Left Foot",
    32: "Right Foot"
}

def calculate_angle(v1, v2):
    """
    Calculate the angle between two vectors using the dot product.
    """
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    angle_radians = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def plot_joint_angles(time, angles, label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=angles, mode='lines', name=label))
    fig.update_layout(title=f"{label} Joint Angles", xaxis_title="Time (s)", yaxis_title="Angle (degrees)")
    st.plotly_chart(fig)

def process_video(video_path, output_txt_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  # Calculate duration in seconds
    if duration > 65:
        st.error(f"Uploaded video duration is {duration:.2f} seconds. Please upload shorter than a 60-second video.")
    else:
        start_time = duration * 0.5 - 0.45 * duration  # Middle start time in seconds 
        end_time = duration * 0.5 + 0.45 * duration    # Middle end time in seconds
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

    st.success(f"Upload successful!")

    # Lists to store joint angles over time
    left_knee_angles, right_knee_angles = [], []
    left_hip_angles, right_hip_angles = [], []
    left_ankle_angles, right_ankle_angles = [], []

    # Initialize MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB (MediaPipe expects RGB images)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame to detect poses
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                def get_coords(landmark):
                    return np.array([landmark.x, landmark.y])

                left_hip = get_coords(landmarks[23])
                right_hip = get_coords(landmarks[24])
                left_knee = get_coords(landmarks[25])
                right_knee = get_coords(landmarks[26])
                left_ankle = get_coords(landmarks[27])
                right_ankle = get_coords(landmarks[28])
                left_foot = get_coords(landmarks[31])
                right_foot = get_coords(landmarks[32])

                # Calculate vectors
                left_thigh_vector = left_hip - left_knee
                left_shank_vector = left_knee - left_ankle
                right_thigh_vector = right_hip - right_knee
                right_shank_vector = right_knee - right_ankle
                left_foot_vector = left_ankle - left_foot
                right_foot_vector = right_ankle - right_foot

                # Calculate joint angles
                left_knee_angles.append(calculate_angle(left_thigh_vector, left_shank_vector))
                right_knee_angles.append(calculate_angle(right_thigh_vector, right_shank_vector))
                left_hip_angles.append(calculate_angle(left_thigh_vector, right_thigh_vector))
                right_hip_angles.append(calculate_angle(right_thigh_vector, left_thigh_vector))
                left_ankle_angles.append(calculate_angle(left_shank_vector, left_foot_vector))
                right_ankle_angles.append(calculate_angle(right_shank_vector, right_foot_vector))
    
    # Save joint angle data to a text file
    with open(output_txt_path, 'w') as file:
        file.write("Time (s), Left Knee, Right Knee, Left Hip, Right Hip, Left Ankle, Right Ankle\n")
        time = np.arange(0, len(left_hip_angles)) / 30  # Time in seconds
        for i in range(len(left_hip_angles)):
            file.write(f"{time[i]:.2f}, {left_knee_angles[i]:.2f}, {right_knee_angles[i]:.2f}, "
                       f"{left_hip_angles[i]:.2f}, {right_hip_angles[i]:.2f}, "
                       f"{left_ankle_angles[i]:.2f}, {right_ankle_angles[i]:.2f}\n")

    st.success(f"Joint angle data saved to {output_txt_path}")

    # Display range of motion for each joint
    st.write("### Range of Motion (Degrees):")
    st.write(f"Left Hip: {max(left_hip_angles) - min(left_hip_angles):.2f}")
    st.write(f"Right Hip: {max(right_hip_angles) - min(right_hip_angles):.2f}")
    st.write(f"Left Knee: {max(left_knee_angles) - min(left_knee_angles):.2f}")
    st.write(f"Right Knee: {max(right_knee_angles) - min(right_knee_angles):.2f}")
    st.write(f"Left Ankle: {max(left_ankle_angles) - min(left_ankle_angles):.2f}")
    st.write(f"Right Ankle: {max(right_ankle_angles) - min(right_ankle_angles):.2f}")

    # Plot joint angles over time
    time = np.arange(0, len(left_hip_angles)) / 30  # Time in seconds
    tick_fontsize = 20

    # st.write('## Hip Angles')
    # fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    # ax[0].plot(time, left_hip_angles, label="Left Hip")
    # ax[1].plot(time, right_hip_angles, label="Right Hip", color='orange')
    # ax[0].legend(fontsize=24)
    # ax[1].legend(fontsize=24)
    # ax[0].set_ylabel("Angle (degrees)", fontsize=20)
    # ax[1].set_xlabel("Time (s)", fontsize=20)
    # ax[1].set_ylabel("Angle (degrees)", fontsize=20)
    # ax[0].set_xlim([time[0], time[-1]])
    # ax[1].set_xlim([time[0], time[-1]])
    # ax[0].tick_params(axis='both', labelsize=tick_fontsize)
    # ax[1].tick_params(axis='both', labelsize=tick_fontsize)
    # st.pyplot(fig)

    # st.write('## Knee Angles')
    # fig2, ax = plt.subplots(2, 1, figsize=(12, 8))
    # ax[0].plot(time, left_knee_angles, label="Left Knee")
    # ax[1].plot(time, right_knee_angles, label="Right Knee", color='orange')
    # ax[0].legend(fontsize=24)
    # ax[1].legend(fontsize=24)
    # ax[0].set_ylabel("Angle (degrees)", fontsize=20)
    # ax[1].set_xlabel("Time (s)", fontsize=20)
    # ax[1].set_ylabel("Angle (degrees)", fontsize=20)
    # ax[0].set_xlim([time[0], time[-1]])
    # ax[1].set_xlim([time[0], time[-1]])
    # ax[0].tick_params(axis='both', labelsize=tick_fontsize)
    # ax[1].tick_params(axis='both', labelsize=tick_fontsize)
    # st.pyplot(fig2)

    # fig3, ax = plt.subplots(2, 1, figsize=(12, 8))
    # ax[0].plot(time, left_ankle_angles, label="Left Ankle")
    # ax[1].plot(time, right_ankle_angles, label="Right Ankle", color='orange')
    # ax[0].legend(fontsize=24)
    # ax[1].legend(fontsize=24)
    # ax[0].set_ylabel("Angle (degrees)", fontsize=20)
    # ax[1].set_xlabel("Time (s)", fontsize=20)
    # ax[1].set_ylabel("Angle (degrees)", fontsize=20)
    # ax[0].set_xlim([time[0], time[-1]])
    # ax[1].set_xlim([time[0], time[-1]])

    # ax[0].tick_params(axis='both', labelsize=tick_fontsize)
    # ax[1].tick_params(axis='both', labelsize=tick_fontsize)
    # st.pyplot(fig3)

    # plot using plotly (replace with process_first_video_frame)
    st.write('## Hip Angles')
    plot_joint_angles(time, left_hip_angles, 'Left Hip')
    plot_joint_angles(time, right_hip_angles, 'Right Hip')
    st.write('## Knee Angles')
    plot_joint_angles(time, left_knee_angles, 'Left Knee')
    plot_joint_angles(time, right_knee_angles, 'Right Knee')
    st.write('## Ankle Angles')
    plot_joint_angles(time, left_ankle_angles, 'Left Ankle')
    plot_joint_angles(time, right_ankle_angles, 'Right Ankle')

    cap.release()

def main():
    st.title("Joint Angle Analysis from Video")
    video_files = st.file_uploader("Upload video(s)", type=["mp4", "avi", "mov"], accept_multiple_files=True)
    if video_files:
        for video_file in video_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:

                temp_video_file.write(video_file.read())
                temp_video_path = temp_video_file.name
                temp_video_file.close()
                output_txt_path = r'/workspaces/PolarPlotter/results/joint_angles.txt'
                process_first_frame(temp_video_path)
                process_video(temp_video_path, output_txt_path)

if __name__ == "__main__":
    main()
