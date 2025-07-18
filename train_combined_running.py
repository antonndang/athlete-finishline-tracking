import torch
from ultralytics import YOLO

# use MPS for M4 Max GPU
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model
model = YOLO('yolov8m.pt')

# Training parameters
data_yaml_path = '/Users/anton/PycharmProjects/100WinnerTracking/RunningDatasets/combined_running_dataset/data.yaml'

results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    device=device,
    workers=4,
    optimizer='Adam',
    lr0=0.001,
    patience=10,
    project='running_detection',
    name='yolov8m_combined',
    exist_ok=True 
)

# Validate the trained model
model.val()

# best trained model
model.export(format='onnx')
