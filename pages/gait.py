import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import tempfile
import os

# Setup MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Keypoints of interest
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
            print("Failed to read the video or the video is empty.")
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
            plt.figure(figsize=(10, 6))
            plt.title("First Frame with Skeleton Overlay")
            plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()
            
            # Filter and display keypoints of interest
            print("Keypoints of Interest:")
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                if id in KEYPOINTS_OF_INTEREST:
                    print(
                        f"{KEYPOINTS_OF_INTEREST[id]} (ID {id}): "
                        f"x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}, visibility={landmark.visibility:.3f}"
                    )
        else:
            print("No pose landmarks detected in the first frame.")
    
    # Release the video capture
    cap.release()

def main():
    print("Please enter the path to your video file (e.g., video.mov):")
    video_path = input("File Path: ").strip()
    
    if video_path and os.path.exists(video_path):
        print(f"Processing video: {video_path}")
        process_first_frame(video_path)
    else:
        print("Invalid file path. Please try again.")

if __name__ == "__main__":
    main()
