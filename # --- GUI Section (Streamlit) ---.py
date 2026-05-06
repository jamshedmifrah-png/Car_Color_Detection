import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.cluster import KMeans
from PIL import Image
import os

# 1. SETUP PAGE (This must be the first Streamlit command)
st.set_page_config(page_title="Nationality Detector", layout="wide")

# 2. HELPER FUNCTIONS
def get_dominant_color(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    # Crop to the chest/dress area
    roi = img[int(height*0.6):height, int(width*0.3):int(width*0.7)]
    pixels = roi.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    color = kmeans.cluster_centers_[0].astype(int)
    return f"RGB {tuple(color)}"

def analyze_person(img_path):
    # Analyze face attributes
    analysis = DeepFace.analyze(img_path, actions=['race', 'emotion', 'age'], enforce_detection=False)[0]
    race = analysis['dominant_race']
    emotion = analysis['dominant_emotion']
    age = analysis['age']
    
    results = {"Nationality/Race": race.capitalize(), "Emotion": emotion.capitalize()}

    # Logic based on your requirements
    if "indian" in race.lower():
        results["Age"] = int(age)
        results["Dress Colour"] = get_dominant_color(img_path)
    elif "white" in race.lower() or "latino" in race.lower():
        results["Nationality/Race"] = "United States"
        results["Age"] = int(age)
    elif "black" in race.lower():
        results["Nationality/Race"] = "African"
        results["Dress Colour"] = get_dominant_color(img_path)
    
    return results

# 3. GUI LAYOUT
st.title("🌍 Nationality & Attribute Detector")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save image locally so DeepFace can read it
    with open("temp_img.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Input Preview", use_container_width=True)
    
    with col2:
        if st.button("Run Analysis"):
            with st.spinner("Analyzing..."):
                try:
                    data = analyze_person("temp_img.jpg")
                    for key, val in data.items():
                        st.metric(label=key, value=val)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")