from ultralytics import YOLO

# Load the YOLOv8x-seg model
model = YOLO('yolov8x-seg.pt')

# Train the model
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    patience=50,
    device=0,
    save=True,
    name='keyboard_detection_seg',
    pretrained=True,
    optimizer='Adam',
    lr0=0.001,
    augment=True
)