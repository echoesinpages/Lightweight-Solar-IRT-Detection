import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from preprocess import AdaptiveMSRCR

st.set_page_config(page_title="Solar Defect Inspector", layout="wide")
st.title("☀️ Lightweight-Solar-IRT-Detection")

# Load the exported ONNX model and the preprocessor
@st.cache_resource
def load_assets():
    model = YOLO("runs/detect/yolo26_local_run/weights/best.onnx", task='detect')
    preprocessor = AdaptiveMSRCR()
    return model, preprocessor

model, preprocessor = load_assets()

uploaded_file = st.file_uploader("Upload a Solar Panel Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with st.spinner("Processing with Adaptive MSRCR & YOLO26..."):
        # 1. Enhance the image
        enhanced_image = preprocessor.enhance(image)
        # 2. Run inference
        results = model.predict(source=enhanced_image, imgsz=640, conf=0.40)
        # 3. Draw bounding boxes
        annotated_frame = results.plot()
        
    with col2:
        st.subheader("AI Defect Analysis")
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        if len(results.boxes) > 0:
            st.error(f"Alert: Found {len(results.boxes)} potential defects.")
        else:
            st.success("Panel is functioning nominally. No defects detected.")
