import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from matplotlib import pyplot as plt

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
    # Calculate dot product
    dot_product = np.dot(v1, v2)
    # Calculate magnitudes
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    # Calculate angle in radians
    angle_radians = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def process_first_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read the video or the video is empty.")
            return
        
        # Convert the frame to RGB (MediaPipe expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect poses
        results = pose.process(frame_rgb)
        
        # Draw the pose landmarks on the frame
        if results.pose_landmarks:
            annotated_frame = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            
            # Display the annotated frame with the skeleton overlay
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Skeleton Overlay")
            # Display keypoints and their values
            st.write("Pose Landmarks (Keypoints):")
            keypoints = []
           
                 # Extract landmarks
            left_hip = results.pose_landmarks.landmark[23]
            right_hip = results.pose_landmarks.landmark[24]
            left_knee = results.pose_landmarks.landmark[25]
            right_knee = results.pose_landmarks.landmark[26]
            left_ankle = results.pose_landmarks.landmark[27]
            right_ankle = results.pose_landmarks.landmark[28]
            left_foot = results.pose_landmarks.landmark[31]
            right_foot = results.pose_landmarks.landmark[32]
            
            # Convert landmarks to numpy arrays
            def get_coords(landmark):
                return np.array([landmark.x, landmark.y, landmark.z])

            left_hip_coords = get_coords(left_hip)
            right_hip_coords = get_coords(right_hip)
            left_knee_coords = get_coords(left_knee)
            right_knee_coords = get_coords(right_knee)
            left_ankle_coords = get_coords(left_ankle)
            right_ankle_coords = get_coords(right_ankle)
            left_foot_coords = get_coords(left_foot)
            right_foot_coords = get_coords(right_foot)

            # Calculate vectors
            left_thigh_vector = left_hip_coords - left_knee_coords
            left_shank_vector = left_knee_coords - left_ankle_coords
            right_thigh_vector = right_hip_coords - right_knee_coords
            right_shank_vector = right_knee_coords - right_ankle_coords
            left_foot_vector = left_ankle_coords - left_foot_coords
            right_foot_vector = right_ankle_coords - right_foot_coords

            # Calculate angles
            left_knee_angle = calculate_angle(left_thigh_vector, left_shank_vector)
            right_knee_angle = calculate_angle(right_thigh_vector, right_shank_vector)
            left_hip_angle = calculate_angle(left_thigh_vector, right_thigh_vector)
            right_hip_angle = calculate_angle(right_thigh_vector, left_thigh_vector)
            left_ankle_angle = calculate_angle(left_shank_vector, left_foot_vector)
            right_ankle_angle = calculate_angle(right_shank_vector, right_foot_vector)

            # Display results
            st.write(f"Left Knee Angle: {left_knee_angle:.2f} degrees")
            st.write(f"Right Knee Angle: {right_knee_angle:.2f} degrees")
            st.write(f"Left Hip Angle: {left_hip_angle:.2f} degrees")
            st.write(f"Right Hip Angle: {right_hip_angle:.2f} degrees")
            st.write(f"Left Ankle Angle: {left_ankle_angle:.2f} degrees")
            st.write(f"Right Ankle Angle: {right_ankle_angle:.2f} degrees")

            for id, landmark in enumerate(results.pose_landmarks.landmark):
                if id in KEYPOINTS_OF_INTEREST:
                    keypoints.append(
                        f"{KEYPOINTS_OF_INTEREST[id]}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}, Visibility: {landmark.visibility:.3f}"
                    )
            st.text("\n".join(keypoints))
        else:
            st.error("No pose landmarks detected in the first frame.")
    
    # Release the video capture
    cap.release()

def main():
    st.title("Pose Estimation on First Frame")
    
    # Upload video file
    uploaded_file = st.file_uploader("Upload a .MOV file", type=["mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mov") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
        
        # Process the first frame of the video
        process_first_frame(temp_path)
        
        # Clean up the temporary file
        os.remove(temp_path)

if __name__ == "__main__":
    main()
