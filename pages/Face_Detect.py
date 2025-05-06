import numpy as np
import cv2 as cv
import joblib
import streamlit as st
from PIL import Image
import time

# Define the function to convert str to bool
def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

# Use default values instead of argparse in Streamlit
face_detection_model = 'model/face_detection_yunet_2023mar.onnx'
face_recognition_model = 'model/face_recognition_sface_2021dec.onnx'
score_threshold = 0.9
nms_threshold = 0.3
top_k = 5000

# Load the face recognition model (SVC)
svc = joblib.load('model/svc.pkl')
mydict = ['LamTuan', 'QuocVuong', 'Thang', 'Tin', 'Tu']

# Function to visualize results
def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            for i in range(5):
                cv.circle(input, (coords[4+i*2], coords[5+i*2]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Streamlit UI
st.title("Face Recognition App")
st.write("This app detects and recognizes faces using your webcam.")

# Layout for buttons in the same row
col1, col2 = st.columns(2)

with col1:
    start_camera = st.button("Start Camera")

with col2:
    stop_camera = st.button("Stop Camera")

if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

if start_camera and not st.session_state.run_camera:
    st.session_state.run_camera = True
    stframe = st.empty()

    # Initialize face detection and recognition
    detector = cv.FaceDetectorYN.create(
        face_detection_model,
        "",
        (320, 320),
        score_threshold,
        nms_threshold,
        top_k
    )
    recognizer = cv.FaceRecognizerSF.create(face_recognition_model, "")

    cap = cv.VideoCapture(0)
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])
    tm = cv.TickMeter()

    while st.session_state.run_camera:
        hasFrame, frame = cap.read()
        if not hasFrame:
            st.write("No frame detected!")
            break

        tm.start()
        faces = detector.detect(frame)
        tm.stop()

        if faces[1] is not None:
            for face in faces[1]:
                score = face[-1]
                if score >= score_threshold:
                    face_align = recognizer.alignCrop(frame, face)
                    face_feature = recognizer.feature(face_align)
                    test_predict = svc.predict(face_feature)
                    result = mydict[test_predict[0]]
                    x, y, w, h = face[:4].astype(int)
                    cv.putText(frame, f"{result} ({score:.2f})", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        visualize(frame, faces, tm.getFPS())

        # Convert BGR to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        # Optional: control frame rate
        time.sleep(0.03)

elif stop_camera and st.session_state.run_camera:
    st.session_state.run_camera = False
    st.write("Camera stopped.")
