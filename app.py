import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# 1. Page Configuration
st.set_page_config(page_title="Car Color & Person Detector", layout="wide")
st.title("🚗 Traffic Signal Monitoring System")
st.markdown("### Internship Task: Car Color Detection & Person Counting")

# 2. Load Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# 3. GUI Sidebar for Upload & Preview
st.sidebar.header("Input Settings")
uploaded_file = st.sidebar.file_uploader("Upload Traffic Video", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Create placeholders for the GUI
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("#### Live Processing Preview")
        video_placeholder = st.empty()
    with col2:
        st.write("#### Signal Statistics")
        stats_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        person_count = 0
        blue_car_count = 0
        other_car_count = 0
        
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            cls = int(result.cls[0])
            conf = float(result.conf[0])

            if conf > 0.4:
                # Logic: Person Detection
                if cls == 0: 
                    person_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

                # Logic: Car Color Detection
                elif cls == 2:
                    # Sample the center of the detection for color
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        avg_color = np.array(cv2.mean(roi)[:3]) # BGR format
                        
                        # Check if Blue channel is dominant
                        if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
                            # GUIDELINE: RED rectangle for BLUE cars
                            rect_color = (0, 0, 255) 
                            label = "Blue Car"
                            blue_car_count += 1
                        else:
                            # GUIDELINE: BLUE rectangle for OTHER colors
                            rect_color = (255, 0, 0) 
                            label = "Other Car"
                            other_car_count += 1
                            
                        cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)
                        cv2.putText(frame, label, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)

        # Update the GUI Statistics
        stats_placeholder.markdown(f"""
        **Current Frame Data:**
        * 🚶 People present: `{person_count}`
        * 🟥 Blue Cars detected: `{blue_car_count}`
        * 🟦 Other Cars detected: `{other_car_count}`
        """)

        # Convert BGR to RGB for Streamlit and update preview
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("Processing Complete!")
    
