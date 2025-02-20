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
from sklearn.decomposition import PCA

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

    frame_number = st.slider(f"Select frame for video ({video_index+1})", 0, total_frames - 1, key=f"frame_{video_index}")

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
def butter_lowpass_filter(data, cutoff=6, fs=30, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

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

        # Store data in DataFrame
    joint_angle_df = pd.DataFrame({
        "Time": filtered_time,
        "Spine": filtered_spine_segment_angles,
        "Left Hip": filtered_left_hip_angles, "Right Hip": filtered_right_hip_angles,
        "Left Knee": filtered_left_knee_angles, "Right Knee": filtered_right_knee_angles,
        "Left Ankle": filtered_left_ankle_angles, "Right Ankle": filtered_right_ankle_angles
    })

    pca_checkbox = st.checkbox("Perform Principle Component Analysis", value=False, key=f"pca_{video_index}")
    if pca_checkbox:
        perform_pca(joint_angle_df)

def perform_pca(df):
    st.write("### Principal Component Analysis (PCA)")

    # Extract numerical joint angle data
    X = df.iloc[:, 1:].values
    
    # User selects number of principal components
    pcs = st.slider('Select the number of Principal Components:', 1, min(30, X.shape[1]), 2)
    st.write(f"Number of Principal Components Selected: {pcs}")
    
    # Perform PCA
    pca = PCA(n_components=pcs)
    principal_components = pca.fit_transform(X)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_ 
    cumulative_variance = np.cumsum(explained_variance) 

    # dataframe for explained variance
    pca_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(len(explained_variance))],
        "Explained Variance (%)": explained_variance * 100,
        "Cumulative Variance (%)": cumulative_variance * 100
    })

    # Get absolute loadings (importance of each feature in each PC)
    loadings = np.abs(pca.components_)

    # Get top contributing features for each PC
    feature_labels = ["Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Spine Angle"]

    top_features_per_pc = []
    for i in range(pcs):
        top_feature_idx = np.argsort(-loadings[i])  # Sort in descending order
        top_features_per_pc.append([feature_labels[j] for j in top_feature_idx])

    # Create DataFrame
    pca_feature_df = pd.DataFrame(top_features_per_pc, index=[f"PC{i+1}" for i in range(pcs)])
    pca_feature_df.columns = [f"Rank {i+1}" for i in range(len(feature_labels))]  # Rank features

        
    top_features_per_pc = []
    for i in range(pcs):
        top_feature_idx = np.argsort(-loadings[i])  # Sort in descending order
        top_features_per_pc.append([feature_labels[j] for j in top_feature_idx])

    # Create DataFrame
    pca_feature_df = pd.DataFrame(top_features_per_pc, 
                                index=[f"PC{i+1}" for i in range(pcs)])
    pca_feature_df.columns = [f"Rank {i+1}" for i in range(len(feature_labels))]  # Rank features
    top_features = pca_feature_df

    fig = go.Figure()
    for i, feature in enumerate(top_features.iloc[:, 0]):  # Use only the top contributing feature
        fig.add_trace(go.Bar(
            x=[f"PC{i+1} ({feature})"],  # Label PC with the top feature
            y=[explained_variance[i] * 100],
            name=f"PC{i+1} ({feature})"
        ))

    explained_variance = pca.explained_variance_ratio_ 
    cumulative_variance = np.cumsum(explained_variance) 
    feature_labels = ["Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Spine Angle"]
    loadings = np.abs(pca.components_)
    top_features_ = [feature_labels[np.argmax(loadings[i])] for i in range(pcs)]

    pca_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1} ({top_features_[i]})" for i in range(len(explained_variance))],
        "Explained Variance (%)": explained_variance * 100,
        "Cumulative Variance (%)": cumulative_variance * 100,
    })

   
    fig.add_trace(go.Scatter(
        x=[f"PC{i+1} ({feature})" for i, feature in enumerate(top_features.iloc[:, 0])],
        y=cumulative_variance * 100,
        mode="lines+markers",
        name="Cumulative Variance (%)",
        line=dict(color='red', dash="dash")
    ))

    fig.update_layout(
        title="Explained Variance with Top Contributing Feature",
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance (%)",
        legend_title="Legend"
    )
    st.plotly_chart(fig)
    # download plot data 
    pca_plot_csv = pca_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download PCA Plot Data",
        data=pca_plot_csv,
        file_name="pca_plot_data.csv",
        mime="text/csv"
    )

    # st.dataframe(top_features)

    st.dataframe(pca_df)

    # combine the two dataframes and download
    pca_data = pd.concat([pca_df, top_features], axis=1)
    pca_feature_csv = pca_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download PCA Data",
        data=pca_feature_csv,
        file_name="pca_data.csv",
        mime="text/csv"
    )
  
    # 2D PCA Scatter Plot (Only if at least 2 PCs are selected)
    if pcs >= 2:
        fig_2d = go.Figure()
        fig_2d.add_trace(go.Scatter(
            x=principal_components[:, 0],
            y=principal_components[:, 1],
            mode='markers',
            marker=dict(size=6, color=df["Time"], colorscale='Viridis'),
            text=df["Time"]
        ))
        fig_2d.update_layout(title="PCA Projection (2D)", xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(fig_2d)
        # download plot button
        pca_2d_csv = pd.DataFrame(principal_components[:, :2], columns=[f"PC{i+1}" for i in range(2)]).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download 2D PCA Data",
            data=pca_2d_csv,
            file_name="pca_2d_data.csv",
            mime="text/csv"
        )

    # 3D PCA Scatter Plot (Only if at least 3 PCs are selected)
    if pcs >= 3:
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=principal_components[:, 0], 
            y=principal_components[:, 1], 
            z=principal_components[:, 2],
            mode='markers', 
            marker=dict(size=4, color=df["Time"], colorscale='Viridis'),
            text=df["Time"]
        )])
        fig_3d.update_layout(title="PCA Projection (3D)")
        st.plotly_chart(fig_3d)
        # download plot button
        pca_3d_csv = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(pcs)]).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download 3D PCA Data",
            data=pca_3d_csv,
            file_name="pca_3d_data.csv",
            mime="text/csv"
        )


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
