import cv2
import mediapipe as mp
import time

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True)

# Function to draw landmarks on the frame
def draw_landmarks(image, landmarks):
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )

# Open video file
video_path = '../videos/demo_01.mp4'
cap = cv2.VideoCapture(video_path)

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect the pose
    results = pose.process(image_rgb)
    
    # Draw the pose annotation on the image
    if results.pose_landmarks:
        draw_landmarks(frame, results.pose_landmarks)
    
    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Dance Pose', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
