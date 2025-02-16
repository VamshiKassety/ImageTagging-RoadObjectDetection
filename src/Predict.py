import os
import logging
from ultralytics import YOLO
import cv2
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
data_dir = 'C:\\Work\\AIML\\DataSets\\BDD_DataSet\\data_dir\\data_dir\\'
VIDEOS_DIR = os.path.join(data_dir, 'predict_videos')
MODEL_PATH = os.path.join(data_dir, 'runs\\detect\\train8\\weights\\best.pt')
VIDEO_NAME = '30sec.mp4'
THRESHOLD = 0.1

# Paths
video_path = os.path.join(VIDEOS_DIR, VIDEO_NAME)
video_path_out = '{}_outt8.mp4'.format(video_path)

# Verify file existence
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Input video not found: {video_path}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model file not found: {MODEL_PATH}")

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Error: Unable to open the video file: {video_path}")

# Get video properties
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize video writer
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), fps, (W, H))

# Load YOLO model
logging.info("Loading YOLO model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(MODEL_PATH).to(device)  # Use GPU if available
logging.info(f"YOLO model loaded successfully on {device} !!!")

# Define colors for bounding boxes
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (192, 192, 192),# Silver
    (255, 165, 0)   # Orange
]

# Process video frames
logging.info(f"Processing video: {video_path}")
for _ in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run YOLO on the frame
    results = model(frame)[0]
    boxes = results.boxes.data.tolist()

    for box in boxes:
        x1, y1, x2, y2, score, class_id = box
        if score > THRESHOLD:
            # Get bounding box color based on class ID
            color = COLORS[int(class_id) % len(COLORS)]
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            
            # Add class name and score as label
            label = f"{results.names[int(class_id)].upper()} {score:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Write the processed frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Final message
logging.info(f"A tracking video has been created with the name: {video_path_out}")
