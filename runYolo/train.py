# Import your custom dataset
from weighted_dataset import YOLOWeightedDataset

# Patch the YOLO dataset class
import ultralytics.data.build as build
build.YOLODataset = YOLOWeightedDataset

# Now proceed with normal YOLO training
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')  # or your pretrained model

# Train the model using your data.yaml
results = model.train(
    data='data.yaml',
    epochs=120,
    imgsz=640,
    # Add other training parameters as needed
)
