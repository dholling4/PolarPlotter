import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Setup MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize a list to store frames with pose landmarks
    processed_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB (MediaPipe expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect poses
        results = pose.process(frame_rgb)
        
        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Convert the frame back to BGR to display in Streamlit
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Store the processed frame for later use
        processed_frames.append(frame_bgr)
    
    cap.release()
    return processed_frames

def display_processed_video(frames):
    # Display processed video frame by frame
    for frame in frames:
        # Convert the frame to RGB for Streamlit display
        st.image(frame, channels="BGR", use_column_width=True)

# Streamlit app
def main():
    st.title("Pose Estimation from .MOV File")
    
    # Upload .MOV file
    uploaded_file = st.file_uploader("Upload a .MOV file", type=["mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        with open("uploaded_video.mov", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("File uploaded successfully! Processing the video...")
        
        # Process the video to extract frames with pose estimation
        frames = process_video("uploaded_video.mov")
        
        # Display processed video with pose landmarks
        display_processed_video(frames)

if __name__ == "__main__":
    main()
