import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.cluster import KMeans
from PIL import Image
import os

# --- 1. Page Config ---
st.set_page_config(page_title="Nationality Detection Project", layout="wide")

# --- 2. Logic Functions ---

def get_dominant_color(img_path):
    """Detects the color of the clothing area."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    # Crop to the chest area (approx bottom 40% center)
    roi = img[int(h*0.6):int(h*0.9), int(w*0.2):int(w*0.8)]
    
    # Simple K-Means to find the most common color
    pixels = roi.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    color = kmeans.cluster_centers_[0].astype(int)
    return f"RGB {tuple(color)}"

def process_nationality_rules(analysis, img_path):
    """Applies the specific rules for Indian, US, African, and others."""
    race = analysis['dominant_race'].lower()
    emotion = analysis['dominant_emotion'].capitalize()
    age = int(analysis['age'])
    
    results = {"Emotion": emotion}
    
    # Rule Mapping
    if "indian" in race:
        results["Nationality"] = "Indian"
        results["Age"] = age
        results["Dress Colour"] = get_dominant_color(img_path)
        
    elif "white" in race or "latino" in race:
        results["Nationality"] = "United States"
        results["Age"] = age
        
    elif "black" in race:
        results["Nationality"] = "African"
        results["Dress Colour"] = get_dominant_color(img_path)
        
    else:
        results["Nationality"] = race.capitalize()
    
    return results

# --- 3. GUI Layout ---

st.title("🌍 Nationality & Attribute Analysis")
st.sidebar.header("Upload Section")

uploaded_file = st.sidebar.file_uploader("Choose an image from UTKFace...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save temp file
    temp_path = "temp_predict.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display Preview
    st.image(uploaded_file, caption="Input Preview", width=400)
    
    if st.button("Predict Nationality & Emotions"):
        with st.spinner("Analyzing weights and image..."):
            try:
                # DeepFace analysis
                analysis_data = DeepFace.analyze(temp_path, actions=['race', 'emotion', 'age'], enforce_detection=False)[0]
                
                # Apply your specific project rules
                final_output = process_nationality_rules(analysis_data, temp_path)
                
                # Show Results
                st.success("Analysis Complete!")
                cols = st.columns(len(final_output))
                for i, (key, val) in enumerate(final_output.items()):
                    cols[i].metric(label=key, value=val)
                    
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload an image in the sidebar to begin.")