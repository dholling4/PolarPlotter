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
from scipy.signal import butter, lfilter
from sklearn.decomposition import NMF

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

# Butterworth lowpass filter functions
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

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
    spine_segment_angles = []
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

                # spine segment angle
                spine_segment_angles.append(calculate_angle(trunk_vector, vertical_vector))                
                left_hip_angles.append(calculate_angle(left_trunk_vector, left_thigh_vector))
                right_hip_angles.append(calculate_angle(right_trunk_vector, right_thigh_vector))
                left_knee_angles.append(calculate_angle(left_thigh_vector, left_shank_vector))
                right_knee_angles.append(calculate_angle(right_thigh_vector, right_shank_vector))
                left_ankle_angles.append(calculate_angle(left_shank_vector, left_foot_vector))
                right_ankle_angles.append(calculate_angle(right_shank_vector, right_foot_vector))

    
    time = np.arange(0, len(left_hip_angles)) / fps  
    cap.release()

    # Apply lowpass filter to smooth angles
    cutoff_frequency = 6  # Adjust cutoff frequency based on signal characteristics
    left_hip_angles = butter_lowpass_filter(left_hip_angles, cutoff_frequency, fps)
    right_hip_angles = butter_lowpass_filter(right_hip_angles, cutoff_frequency, fps)
    left_knee_angles = butter_lowpass_filter(left_knee_angles, cutoff_frequency, fps)
    right_knee_angles = butter_lowpass_filter(right_knee_angles, cutoff_frequency, fps)
    left_ankle_angles = butter_lowpass_filter(left_ankle_angles, cutoff_frequency, fps)
    right_ankle_angles = butter_lowpass_filter(right_ankle_angles, cutoff_frequency, fps)
    spine_segment_angles = butter_lowpass_filter(spine_segment_angles, cutoff_frequency, fps) 

        ### CROP HERE ###
    start_time, end_time = st.slider("Select time range", min_value=float(0), max_value=float(time[-1]), value=(float(0), float(time[-1])))
    st.write(f"Selected frame range: {start_frame} to {end_frame}")
    st.write(f"Selected time range: {start_time:.2f}s to {end_time:.2f}s")
    mask = (time >= start_time) & (time <= end_time)
    filtered_time = time[mask]

    filtered_spine_segment_angles = np.array(spine_segment_angles)[mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_time, y=filtered_spine_segment_angles, mode='lines', name="Spine Segment Angles"))
    fig.add_trace(go.Scatter(x=[frame_time, frame_time], y=[min(filtered_spine_segment_angles), max(filtered_spine_segment_angles)], mode='lines', line=dict(color='red', dash='dash'), name='Selected Frame'))
    fig.update_layout(title=f"Spine Segment Angles", xaxis_title="Time (s)", yaxis_title="Angle (degrees)")
    st.plotly_chart(fig)

    # Assuming filtered_time and filtered_spine_segment_angles are lists or numpy arrays
    spine_data = {
        "Time (s)": filtered_time,
        "Spine Segment Angles (degrees)": filtered_spine_segment_angles
    }

    # Create a DataFrame
    spine_df = pd.DataFrame(spine_data)

    # Convert DataFrame to CSV
    spine_csv = spine_df.to_csv(index=False).encode('utf-8')

                # Add download csv button
    st.download_button(
    label="Download Spine Segment Angle Data",
    data=spine_csv,
    file_name="spine_segment_angles.csv",
    mime="text/csv"
)

    filtered_left_hip_angles = np.array(left_hip_angles)[mask]
    filtered_right_hip_angles = np.array(right_hip_angles)[mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_time, y=filtered_left_hip_angles, mode='lines', name="Left Hip"))
    fig.add_trace(go.Scatter(x=filtered_time, y=filtered_right_hip_angles, mode='lines', name="Right Hip"))
    fig.add_trace(go.Scatter(x=[frame_time, frame_time], y=[min(np.min(filtered_left_hip_angles), np.min(filtered_right_hip_angles)), max(np.max(filtered_left_hip_angles), np.max(filtered_left_hip_angles))], mode='lines', line=dict(color='red', dash='dash'), name='Selected Frame'))
    fig.update_layout(title=f"Hip Joint Angles", xaxis_title="Time (s)", yaxis_title="Angle (degrees)")
    st.plotly_chart(fig)

    hip_data = {
        "Time (s)": filtered_time,
        "Left Hip Angle (degrees)": filtered_left_hip_angles,
        "Right Hip Angle (degrees)": filtered_right_hip_angles
    }

    # Create a DataFrame
    hip_df = pd.DataFrame(hip_data)

    # Convert DataFrame to CSV
    hip_csv = hip_df.to_csv(index=False).encode('utf-8')

    # Add download csv button
    st.download_button(
        label="Download Hip Angle Data",
        data=hip_csv,
        file_name="hip_angles.csv",
        mime="text/csv"
    )
    
    filtered_left_knee_angles = np.array(left_knee_angles)[mask]
    filtered_right_knee_angles = np.array(right_knee_angles)[mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_time, y=filtered_left_knee_angles, mode='lines', name="Left Knee"))
    fig.add_trace(go.Scatter(x=filtered_time, y=filtered_right_knee_angles, mode='lines', name="Right Knee"))
    fig.add_trace(go.Scatter(x=[frame_time, frame_time], y=[min(np.min(filtered_left_knee_angles), np.min(filtered_right_knee_angles)), max(np.max(filtered_left_knee_angles), np.max(filtered_left_knee_angles))], mode='lines', line=dict(color='red', dash='dash'), name='Selected Frame'))
    fig.update_layout(title=f"Knee Joint Angles", xaxis_title="Time (s)", yaxis_title="Angle (degrees)")
    st.plotly_chart(fig)

    knee_data = {
        "Time (s)": filtered_time,
        "Left Knee Angle (degrees)": filtered_left_knee_angles,
        "Right Knee Angle (degrees)": filtered_right_knee_angles
    }

    # Create a DataFrame
    knee_df = pd.DataFrame(knee_data)

    # Convert DataFrame to CSV
    knee_csv = knee_df.to_csv(index=False).encode('utf-8')

    # Add download csv button
    st.download_button(
        label="Download Knee Angle Data",
        data=knee_csv,
        file_name="knee_angles.csv",
        mime="text/csv"
    )

    filtered_left_ankle_angles = np.array(left_ankle_angles)[mask]
    filtered_right_ankle_angles = np.array(right_ankle_angles)[mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_time, y=filtered_left_ankle_angles, mode='lines', name="Left Ankle"))
    fig.add_trace(go.Scatter(x=filtered_time, y=filtered_right_ankle_angles, mode='lines', name="Right Ankle"))
    fig.add_trace(go.Scatter(x=[frame_time, frame_time], y=[min(np.min(filtered_left_ankle_angles), np.min(filtered_right_ankle_angles)), max(np.max(filtered_left_ankle_angles), np.max(filtered_left_ankle_angles))], mode='lines', line=dict(color='red', dash='dash'), name='Selected Frame'))
    fig.update_layout(title=f"Ankle Joint Angles", xaxis_title="Time (s)", yaxis_title="Angle (degrees)")
    st.plotly_chart(fig)
    
    ankle_data = {
        "Time (s)": filtered_time,
        "Left Ankle Angle (degrees)": filtered_left_ankle_angles,
        "Right Ankle Angle (degrees)": filtered_right_ankle_angles
    }

    # Create a DataFrame
    ankle_df = pd.DataFrame(ankle_data)

    # Convert DataFrame to CSV
    ankle_csv = ankle_df.to_csv(index=False).encode('utf-8')

    # Add download csv button
    st.download_button(
        label="Download Ankle Angle Data",
        data=ankle_csv,
        file_name="ankle_angles.csv",
        mime="text/csv"
    )     


    ### END CROP ###

  # show tables
    df = pd.DataFrame({'Time': time, 'Spine Segment Angles': spine_segment_angles, 'Left Joint Hip': left_hip_angles, 'Right Hip': right_hip_angles, 'Left Knee': left_knee_angles, 'Right Knee': right_knee_angles, 'Left Ankle': left_ankle_angles, 'Right Ankle': right_ankle_angles})
    st.write('### Joint Angles (deg)')

    st.dataframe(df)

    st.write('### Range of Motion')
    # create dataframe of range of motion
    
    df_rom = pd.DataFrame({'Joint': ['Spine Segment Angle', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'], 'Range of Motion (degrees)': [np.ptp(spine_segment_angles), np.ptp(left_hip_angles), np.ptp(right_hip_angles), np.ptp(left_knee_angles), np.ptp(right_knee_angles), np.ptp(left_ankle_angles), np.ptp(right_ankle_angles)]})
    # add columns for the min and max angles for each joint
    df_rom['Min Angle (degrees)'] = [np.min(spine_segment_angles), np.min(left_hip_angles), np.min(right_hip_angles), np.min(left_knee_angles), np.min(right_knee_angles), np.min(left_ankle_angles), np.min(right_ankle_angles)]
    df_rom['Max Angle (degrees)'] = [np.max(spine_segment_angles), np.max(left_hip_angles), np.max(right_hip_angles), np.max(left_knee_angles), np.max(right_knee_angles), np.max(left_ankle_angles), np.max(right_ankle_angles)]
    df_rom.columns = ['Joint', 'Min Angle (deg)', 'Max Angle (deg)', 'Range of Motion (deg)',]
    st.dataframe(df_rom)

    # show the range of motion as a spider plot
    fig = go.Figure()
    
    rom_values = [
    np.ptp(right_knee_angles),
    np.ptp(right_hip_angles),
    np.ptp(spine_segment_angles),
    np.ptp(left_hip_angles),
    np.ptp(left_knee_angles),
    np.ptp(left_ankle_angles),
    np.ptp(right_ankle_angles)
        ]
    
    joint_labels = ['Right Joint Knee', 'Right Joint Hip', 'Spine Segment Angle', 'Left Joint Hip', 'Left Joint Knee', 'Left Joint Ankle', 'Right Joint Ankle']

    fig.add_trace(go.Scatterpolar(
        r=[np.ptp(right_knee_angles), np.ptp(right_hip_angles), np.ptp(spine_segment_angles), np.ptp(left_hip_angles), np.ptp(left_knee_angles), np.ptp(left_ankle_angles), np.ptp(right_ankle_angles)],
        theta=joint_labels,
        fill='toself',
        name='Range of Motion'
    ))

    max_all_joint_angles = np.max([np.ptp(right_knee_angles), np.ptp(right_hip_angles), np.ptp(spine_segment_angles), np.ptp(left_hip_angles), np.ptp(left_knee_angles), np.ptp(left_ankle_angles), np.ptp(right_ankle_angles)])
    
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

    synergy = st.checkbox("Perform Synergy Analysis", value=False, key=f"synergy_{video_index}")

    if synergy:
        # Simulated kinematic data (timepoints x joints)
        np.random.seed(42)

        # Select number of synergies
        n_synergies = st.slider("Select the Number of Synergies", 1, 10, 4, key=f"synergy_slider_{video_index}")

        # Combine all joint angles into a single matrix (timepoints x joints)
        kinematic_data = np.column_stack([
            filtered_spine_segment_angles,
            filtered_left_hip_angles,
            filtered_left_knee_angles,
            filtered_left_ankle_angles
        ])

        # Apply NMF to the full kinematic dataset
        nmf = NMF(n_components=n_synergies, init='random', random_state=42)
        W = nmf.fit_transform(kinematic_data)  # Synergy activations over time
        H = nmf.components_  # Feature contributions per synergy

        # Plot extracted movement synergies for all joints
        joint_names = ["Spine", "Hip", "Knee", "Ankle"]
        
        # for j, joint in enumerate(joint_names):
        fig = go.Figure()
        for i in range(n_synergies):
            fig.add_trace(go.Scatter(
                x=filtered_time, 
                y=W[:, i], 
                mode='lines', 
                name=f'Synergy {i+1}'
            ))
        fig.update_layout(title=f'Extracted Joint Synergies', xaxis_title='Time', yaxis_title='Activation')
        st.plotly_chart(fig)

        df_synergy = pd.DataFrame(H, columns=joint_names, index=[f'Synergy {i+1}' for i in range(n_synergies)])
        # st.write('### Synergy Feature Contributions')
        st.dataframe(df_synergy)
        # download button
        synergy_csv = df_synergy.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download Synergy Feature Contributions",
            data=synergy_csv,
            file_name="synergy_feature_contributions.csv",
            mime="text/csv")

        # Plot H as a heatmap
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=H, 
            x=joint_names, 
            y=[f'Synergy {i+1}' for i in range(n_synergies)], 
            colorscale='Viridis'
        ))
        fig.update_layout(title='Feature Contributions in Each Synergy (H Matrix)', xaxis_title='Joint Angles', yaxis_title='Synergies')

        st.plotly_chart(fig)
        # download heatmap
        synergy_heatmap_csv = pd.DataFrame(H, columns=joint_names, index=[f'Synergy {i+1}' for i in range(n_synergies)]).to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download Synergy Heatmap",
            data=synergy_heatmap_csv,
            file_name="synergy_heatmap.csv",
            mime="text/csv")

    ## finish code here
    # synergy = st.checkbox("Perform Synergy Analysis", value=False, key=f"synergy_{video_index}")
    # if synergy:
    #     # Simulated kinematic data (timepoints x joints)
    #     np.random.seed(42)

    #     # Apply NMF
    #     n_synergies = st.slider("Select the Number of Synergies", 1, 10, 4, key=f"synergy_slider_{video_index}")
        
    #     nmf = NMF(n_components=n_synergies, init='random', random_state=42)
    #     W = nmf.fit_transform(filtered_spine_segment_angles.reshape(-1,1))  # Synergy patterns
    #     W_hip = nmf.fit_transform(filtered_left_hip_angles.reshape(-1,1))
    #     W_knee = nmf.fit_transform(filtered_left_knee_angles.reshape(-1,1))
    #     W_ankle = nmf.fit_transform(filtered_left_ankle_angles.reshape(-1,1))
        
    #     H = nmf.components_  # Activation patterns

    #     # Plot the extracted movement synergies
    #     fig = go.Figure()
    #     for i in range(n_synergies):
    #         fig.add_trace(go.Scatter(x=filtered_time, y=W[:, i], mode='lines', name=f'Synergy {i+1}'))
    #     fig.update_layout(title='Extracted Synergies (Spine Segment Angle)', xaxis_title='Time', yaxis_title='Activation')
    #     st.plotly_chart(fig)

    #     fig = go.Figure()
    #     for i in range(n_synergies):
    #         fig.add_trace(go.Scatter(x=filtered_time, y=W_hip[:, i], mode='lines', name=f'Synergy {i+1}'))
    #     fig.update_layout(title='Extracted Hip Synergies', xaxis_title='Time', yaxis_title='Activation')
    #     st.plotly_chart(fig)

    #     fig = go.Figure()
    #     for i in range(n_synergies):
    #         fig.add_trace(go.Scatter
    #         (x=filtered_time, y=W_knee[:, i], mode='lines', name=f'Synergy {i+1}'))
    #     fig.update_layout(title='Extracted Knee Synergies', xaxis_title='Time', yaxis_title='Activation')
    #     st.plotly_chart(fig)

    #     fig = go.Figure()
    #     for i in range(n_synergies):
    #         fig.add_trace(go.Scatter
    #         (x=filtered_time, y=W_ankle[:, i], mode='lines', name=f'Synergy {i+1}'))
    #     fig.update_layout(title='Extracted Ankle Synergies', xaxis_title='Time', yaxis_title='Activation')
    #     st.plotly_chart(fig)

    #     # plot H as the heatmap 
    #     fig = go.Figure()
    #     fig.add_trace(go.Heatmap(z=H, x=['Spine', 'Hip', 'Knee', 'Ankle'], y=[f'Synergy {i+1}' for i in range(n_synergies)], colorscale='Viridis'))
    #     fig.update_layout(title='Feature Contributions in Each Synergy (H Matrix)', xaxis_title='Joint Angles', yaxis_title='Synergies')

    #     st.plotly_chart(fig)
               
       

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
