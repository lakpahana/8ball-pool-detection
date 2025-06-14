from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.predict("ssss.jpg",save=True)