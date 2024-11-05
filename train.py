import torch
from ultralytics import YOLO

# Check if a GPU is available
if torch.cuda.is_available():
    # Print GPU details
    device_name = torch.cuda.get_device_name(0)  # Check first GPU (GPU 0)
    print(f"Training on GPU: {device_name}")

    # Load the YOLOv8 model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Train the model on the specified GPU
    results = model.train(data="data.yaml", epochs=100, device=0)
else:
    print("No GPU detected. Exiting...")
    exit()  # or handle as needed if no GPU is found
