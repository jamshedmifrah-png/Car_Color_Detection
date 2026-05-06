from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load YOLO model
model = YOLO("yolov8n.pt")

# Detect dominant color
def get_dominant_color(image):
    if image.size == 0:
        return (0, 0, 0)

    img = cv2.resize(image, (50, 50))
    img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(img)

    return kmeans.cluster_centers_[0]

# Check if car is blue
def is_blue(color):
    b, g, r = color
    return b > 100 and b > g and b > r

# Main processing function
def process_frame(frame):
    results = model(frame)

    car_count = 0
    people_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            if label == "car":
                car_count += 1
                color = get_dominant_color(roi)

                if is_blue(color):
                    box_color = (0, 0, 255)  # Red box
                else:
                    box_color = (255, 0, 0)  # Blue box

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            elif label == "person":
                people_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, car_count, people_count