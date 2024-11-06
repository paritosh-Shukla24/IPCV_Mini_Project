from ultralytics import YOLO
import cv2
import torch
from IPython.display import display, Image  # For displaying images in Jupyter

# Load the trained model
model = YOLO('best_pose.pt')  # Update path to your model

# Move the model to the desired device (use 'cpu' if CUDA is not available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Define the video path
video_path = '85.mp4'  # Replace with the path to your video file

# Open the video using OpenCV
cap = cv2.VideoCapture(video_path)

# Get video properties (frame width, height, and fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output
output_path = f'{video_path}output_video_pose.mp4'  # Define your output path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (width // 2, height // 2))  # Save at half size

# Loop through each frame in the video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no more frames

    # Resize frame to half size to reduce memory consumption
    frame = cv2.resize(frame, (width // 2, height // 2))

    # Run the frame through the model for pose detection
    results = model(frame)

    # Extract and visualize the poses on the frame
    annotated_frame = results[0].plot()  # Plotting keypoints and skeletons on the frame

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display every 50th frame in Jupyter (optional, comment this out to avoid display)
    if frame_count % 50 == 0:  # Adjust frame display frequency as needed
        _, encoded_img = cv2.imencode('.png', annotated_frame)  # Encode as PNG
        display(Image(data=encoded_img.tobytes()))  # Display frame in Jupyter

    # Release GPU memory after each frame to prevent memory overload
    torch.cuda.empty_cache()

    frame_count += 1

# Release video resources
cap.release()
out.release()

print("Video processing complete. Output saved as", output_path)
