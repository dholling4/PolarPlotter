import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os
import moviepy.editor as m
# Setup MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# import moviepy.editor at mp
        
def convert_mov_to_mp4(input_path, output_path):
    clip = mp.VideoFileClip(input_path)
    clip.write_videofile(output_path, codec="libx264")


def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the video properties (frame rate, width, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a temporary file to save the processed video
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output_path = temp_output_file.name
    temp_output_file.close()  # Close the file so OpenCV can write to it
    
    # Create a video writer to save the processed frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
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
        
        # Write the frame with the pose landmarks drawn to the video file
        out.write(frame)
    
    cap.release()
    out.release()
    
    return temp_output_path

def main():
    st.title("Pose Estimation from .MOV File with Skeleton Overlay")
    
    # Upload .MOV file
    uploaded_file = st.file_uploader("Upload a .MOV file", type=["mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("uploaded_video.mov", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("File uploaded successfully! Processing the video...")
        
        # Process the video and get the path to the processed video
        processed_video_path = process_video("uploaded_video.mov")
        
        # Stream the processed video with pose landmarks
        st.video(processed_video_path)

        # Optionally, cleanup the temporary file
        os.remove(processed_video_path)

if __name__ == "__main__":
    main()
