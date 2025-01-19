import streamlit as st
import cv2
import mediapipe as mp
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
    31: "Left Foot",
    32: "Right Foot"
}

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
