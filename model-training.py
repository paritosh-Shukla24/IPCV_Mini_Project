from ultralytics import YOLO

# Load the pre-trained pose model
model = YOLO('yolov8n-pose.pt')

# Train the model
model.train(
    data='data2.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='yolov8-pose-shoplift',
    device=0
)
