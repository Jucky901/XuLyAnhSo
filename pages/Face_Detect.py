import numpy as np
import cv2 as cv
import joblib
import streamlit as st
import time
import tempfile

# Load the face recognition model (SVC)
svc = joblib.load('model/svc.pkl')
mydict = ['LamTuan', 'QuocVuong', 'Thang', 'Tin', 'Tu']

# Model paths
face_detection_model = 'model/face_detection_yunet_2023mar.onnx'
face_recognition_model = 'model/face_recognition_sface_2021dec.onnx'

# Parameters
score_threshold = 0.8
nms_threshold = 0.3
top_k = 5000
input_width, input_height = 480, 480  # Input size expected by the model
target_fps = 20

# Function to visualize results
def visualize(input_frame, faces, fps, thickness=2):
    if faces[1] is not None:
        for face in faces[1]:
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input_frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            for i in range(5):
                cv.circle(input_frame, (coords[4+i*2], coords[5+i*2]), 2, (0, 255, 255), thickness)
    cv.putText(input_frame, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Streamlit UI
st.title("Optimized Video and Camera Face Recognition")
mode = st.selectbox("Select Mode", ("Video Upload", "Real-Time Camera"))

if mode == "Video Upload":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        cap = cv.VideoCapture(temp_file_path)

elif mode == "Real-Time Camera":
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, input_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, input_height)

if 'cap' in locals() and cap.isOpened():
    frame_fps = cap.get(cv.CAP_PROP_FPS)
    st.write(f"Original FPS: {frame_fps}")
    stframe = st.empty()
    tm = cv.TickMeter()

    # Initialize face detection and recognition
    detector = cv.FaceDetectorYN.create(
        face_detection_model,
        "",
        (input_width, input_height),
        score_threshold,
        nms_threshold,
        top_k
    )
    recognizer = cv.FaceRecognizerSF.create(face_recognition_model, "")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, (input_width, input_height))

        tm.start()
        faces = detector.detect(frame)
        tm.stop()

        fps = tm.getFPS()
        tm.reset()

        if faces[1] is not None:
            for face in faces[1]:
                score = face[-1]
                if score >= score_threshold:
                    face_align = recognizer.alignCrop(frame, face)
                    face_feature = recognizer.feature(face_align)
                    test_predict = svc.predict(face_feature)
                    result = mydict[test_predict[0]]
                    x, y, w, h = face[:4].astype(int)
                    cv.putText(frame, f"{result} ({score:.2f})", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        visualize(frame, faces, fps)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        processing_delay = max(1.0 / target_fps - tm.getTimeSec(), 0)
        time.sleep(processing_delay)

    cap.release()
    st.write("Processing completed.")
