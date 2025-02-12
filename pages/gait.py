import streamlit as st
import cv2
import mediapipe as mp
from mediapipe import solutions
import numpy as np
import tempfile
import os
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import pandas as pd

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

def process_first_frame(video_path, video_index):

    neon_green = (57, 255, 20)
    cool_blue = (0, 91, 255)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    st.write(f"Total frames: {total_frames}, FPS: {fps:.1f}, Duration: {duration:.2f} seconds")

    frame_number = st.slider(f"Select frame ({video_index+1})", 0, total_frames - 1, key=f"frame_{video_index}")

    time = frame_number / fps

    st.write(f'Frame Number:  {frame_number} | Time :  {time:.2f} sec')
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
                annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=solutions.drawing_styles.DrawingSpec(color=neon_green, thickness=10, circle_radius=7),
            connection_drawing_spec=solutions.drawing_styles.DrawingSpec(color=cool_blue, thickness=10)
            )
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_number}")
    cap.release()
    return frame_number, time

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

###
def plot_joint_angles(time, angles, label, frame_time):
    fig = go.Figure()
    
    # Add the joint angle curve
    fig.add_trace(go.Scatter(x=time, y=angles, mode='lines', name=label))
    
    # Add vertical line for selected frame
    fig.add_trace(go.Scatter(
        x=[frame_time, frame_time],
        y=[min(angles), max(angles)],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Selected Frame'
    ))
    
    fig.update_layout(
        title=f"{label} Joint Angles",
        xaxis_title="Time (s)",
        yaxis_title="Angle (degrees)"
    )
    
    st.plotly_chart(fig)

def process_video(video_path, output_txt_path, frame_time, video_index):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  
    
    start_frame = 0
    end_frame = total_frames
    
    left_knee_angles, right_knee_angles = [], []
    left_hip_angles, right_hip_angles = [], []
    left_ankle_angles, right_ankle_angles = [], []
    spine_flexion_angles = []
    thorax_angles, lumbar_angles = [], []


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                def get_coords(landmark):
                    return np.array([landmark.x, landmark.y])
                
                left_shoulder = get_coords(landmarks[11])
                right_shoulder = get_coords(landmarks[12])                 
                
                left_hip = get_coords(landmarks[23])
                right_hip = get_coords(landmarks[24])
                left_knee = get_coords(landmarks[25])
                right_knee = get_coords(landmarks[26])
                left_ankle = get_coords(landmarks[27])
                right_ankle = get_coords(landmarks[28])
                left_foot = get_coords(landmarks[31])
                right_foot = get_coords(landmarks[32])

                # midpoint of trunk vector
                shoulder_mid = (left_shoulder + right_shoulder) / 2
                hip_mid = (left_hip + right_hip) / 2
                
                trunk_height = np.linalg.norm(shoulder_mid - hip_mid)  # Euclidean distance between shoulder and hip midpoints
                # Estimate C7 and T10 positions as percentages of trunk height
                c7_offset = trunk_height *0.18  # C7 is approximately 18% from the top of the trunk
                t10_offset = trunk_height * 0.50  # T10 is approximately 50% from the top of the trunk

                # C7 and T10 coordinates based on the midpoint positions and offsets
                c7_vector = shoulder_mid - np.array([0, c7_offset])
                t10_vector = shoulder_mid - np.array([0, t10_offset])
                thorax_vector = c7_vector - hip_mid
                lumbar_vector = t10_vector - hip_mid

                trunk_vector = shoulder_mid - hip_mid

                # Upward vertical in image coordinates
                vertical_vector = np.array([0, -1])  
                left_trunk_vector = left_shoulder - left_hip
                right_trunk_vector = right_shoulder - right_hip
                left_thigh_vector = left_hip - left_knee
                left_shank_vector = left_knee - left_ankle
                right_thigh_vector = right_hip - right_knee
                right_shank_vector = right_knee - right_ankle
                left_foot_vector = left_ankle - left_foot
                right_foot_vector = right_ankle - right_foot

                thorax_angles.append(calculate_angle(thorax_vector, lumbar_vector))
                lumbar_angles.append(calculate_angle(lumbar_vector, vertical_vector))

                spine_flexion_angles.append(calculate_angle(trunk_vector, vertical_vector))                
                left_hip_angles.append(calculate_angle(left_trunk_vector, left_thigh_vector))
                right_hip_angles.append(calculate_angle(right_trunk_vector, right_thigh_vector))
                left_knee_angles.append(calculate_angle(left_thigh_vector, left_shank_vector))
                right_knee_angles.append(calculate_angle(right_thigh_vector, right_shank_vector))
                left_ankle_angles.append(calculate_angle(left_shank_vector, left_foot_vector))
                right_ankle_angles.append(calculate_angle(right_shank_vector, right_foot_vector))
    
    time = np.arange(0, len(left_hip_angles)) / fps  
    cap.release()

    st.write('Check the boxes below to plot the joint angles.')

    plot_spine_flexion_angles = st.checkbox('Spine Flexion', value=False, key=f'spine_flexion_{video_index}')
    if plot_spine_flexion_angles:
        st.write('### Thorax, Lumbar, and Spine Flexion Angles')
        fig = go.Figure()

        # Combine thorax, lumbar, and spine flexion angles on a single plot
        fig.add_trace(go.Scatter(x=time, y=thorax_angles, mode='lines', name='Thorax'))
        fig.add_trace(go.Scatter(x=time, y=lumbar_angles, mode='lines', name='Lumbar'))
        fig.add_trace(go.Scatter(x=time, y=spine_flexion_angles, mode='lines', name='Spine Flexion'))
        
        fig.update_layout(
            title="Thorax, Lumbar, and Spine Flexion Angles Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Angle (degrees)"
        )
        
        st.plotly_chart(fig)

    hip_angles = st.checkbox('Hip Angles', value=False, key=f'hip_angles_{video_index}')
    if hip_angles:
        st.write('### Hip Angles')
        plot_joint_angles(time, left_hip_angles, 'Left Hip', frame_time)
        plot_joint_angles(time, right_hip_angles, 'Right Hip', frame_time)

    knee_angles = st.checkbox('Knee Angles', value=False, key=f'knee_angles_{video_index}')
    if knee_angles:
        st.write('### Knee Angles')
        plot_joint_angles(time, left_knee_angles, 'Left Knee', frame_time)
        plot_joint_angles(time, right_knee_angles, 'Right Knee', frame_time)

    ankle_angles = st.checkbox('Ankle Angles', value=False, key=f'ankle_angles_{video_index}')
    if ankle_angles:
        st.write('### Ankle Angles')
        plot_joint_angles(time, left_ankle_angles, 'Left Ankle', frame_time)
        plot_joint_angles(time, right_ankle_angles, 'Right Ankle', frame_time)

  # show tables
    df = pd.DataFrame({'Time': time, 'Spine': spine_flexion_angles, 'Left Hip': left_hip_angles, 'Right Hip': right_hip_angles, 'Left Knee': left_knee_angles, 'Right Knee': right_knee_angles, 'Left Ankle': left_ankle_angles, 'Right Ankle': right_ankle_angles})
    st.write('### Joint Angles (deg)')

    st.dataframe(df)

    st.write('### Range of Motion')
    # create dataframe of range of motion
    
    df_rom = pd.DataFrame({'Joint': ['Spine', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'], 'Range of Motion (degrees)': [np.ptp(spine_flexion_angles), np.ptp(left_hip_angles), np.ptp(right_hip_angles), np.ptp(left_knee_angles), np.ptp(right_knee_angles), np.ptp(left_ankle_angles), np.ptp(right_ankle_angles)]})
    # add columns for the min and max angles for each joint
    df_rom['Min Angle (degrees)'] = [np.min(spine_flexion_angles), np.min(left_hip_angles), np.min(right_hip_angles), np.min(left_knee_angles), np.min(right_knee_angles), np.min(left_ankle_angles), np.min(right_ankle_angles)]
    df_rom['Max Angle (degrees)'] = [np.max(spine_flexion_angles), np.max(left_hip_angles), np.max(right_hip_angles), np.max(left_knee_angles), np.max(right_knee_angles), np.max(left_ankle_angles), np.max(right_ankle_angles)]
    df_rom.columns = ['Joint', 'Min Angle (deg)', 'Max Angle (deg)', 'Range of Motion (deg)',]
    st.dataframe(df_rom)

    # show the range of motion as a spider plot
    fig = go.Figure()
    
    rom_values = [
    np.ptp(right_knee_angles),
    np.ptp(right_hip_angles),
    np.ptp(spine_flexion_angles),
    np.ptp(left_hip_angles),
    np.ptp(left_knee_angles),
    np.ptp(left_ankle_angles),
    np.ptp(right_ankle_angles)
        ]
    
    joint_labels = ['Right Knee', 'Right Hip', 'Spine Flexion', 'Left Hip', 'Left Knee', 'Left Ankle', 'Right Ankle']

    fig.add_trace(go.Scatterpolar(
        r=[np.ptp(right_knee_angles), np.ptp(right_hip_angles), np.ptp(spine_flexion_angles), np.ptp(left_hip_angles), np.ptp(left_knee_angles), np.ptp(left_ankle_angles), np.ptp(right_ankle_angles)],
        theta=joint_labels,
        fill='toself',
        name='Range of Motion'
    ))

    max_all_joint_angles = np.max([np.ptp(right_knee_angles), np.ptp(right_hip_angles), np.ptp(spine_flexion_angles), np.ptp(left_hip_angles), np.ptp(left_knee_angles), np.ptp(left_ankle_angles), np.ptp(right_ankle_angles)])
    
    # Add annotations for each data point (ROM value)
    annotations = []
    for i, value in enumerate(rom_values):
        annotations.append(
            dict(
                r=value + 1,  # Slightly offset for visibility
                theta=joint_labels[i],
                text=f"{value:.1f}°",  # Display ROM value with 1 decimal places
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20,
                font=dict(size=12, color="black")
            )
        )
    fig.update_layout(
        title="Range of Motion",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_all_joint_angles + 5],
                tickfont=dict(color='black')  # Set black font for tick values
            )),
        # annotations=annotations,
        showlegend=False
    )
    st.plotly_chart(fig)
       

def main():
    st.title("Joint Angle Analysis from Video")
    video_files = st.file_uploader("Upload video(s)", type=["mp4", "avi", "mov"], accept_multiple_files=True)
    if video_files:
        for idx, video_file in enumerate(video_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(video_file.read())
                temp_video_path = temp_video_file.name
                temp_video_file.close()
                output_txt_path = r'/workspaces/PolarPlotter/results/joint_angles.txt'
                frame_number, frame_time = process_first_frame(temp_video_path, video_index=idx)
                process_video(temp_video_path, output_txt_path, frame_time, video_index=idx)

if __name__ == "__main__":
    main()
