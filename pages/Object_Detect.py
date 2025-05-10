import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import tempfile

# Load the model once
model = YOLO('model/best.pt')

st.set_page_config(page_title="YOLO Image Predictor", layout="centered")
st.title("Object Detection:")
st.write("Ứng dụng có thể phát hiện 5 vật thể sau: Chuối, Sầu Riêng, Táo, Thanh Long và Xoài")

# Upload file
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "bmp", "webp"])

if uploaded_file:
    # Read uploaded file as OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display original image
    st.subheader("Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

    # Predict button
    if st.button("Predict with YOLO"):
        names = model.names
        imgout = image.copy()
        annotator = Annotator(imgout)

        results = model.predict(imgout, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.tolist()

        for box, cls, conf in zip(boxes, clss, confs):
            label = f"{names[int(cls)]} {conf:.2f}"
            annotator.box_label(box, label=label, txt_color=(255, 0, 0), color=(255, 255, 255))

        # Show output image
        st.subheader("Predicted Image")
        st.image(cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB), channels="RGB")
