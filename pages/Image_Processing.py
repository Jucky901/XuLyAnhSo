import streamlit as st
import numpy as np
import cv2
from PIL import Image

import ImageProcessing.Chapter03 as c3
import ImageProcessing.Chapter04 as c4
import ImageProcessing.Chapter05 as c5
import ImageProcessing.Chapter09 as c9

# Cấu hình trang Streamlit
def setup_page():
    st.set_page_config(page_title="Xử lý ảnh số", page_icon="🌅")
    st.markdown("# Xử lý ảnh")
    st.write()

# Tải lên ảnh
def upload_image():
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg', 'tif'])
    
    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        imgin_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE) 
        imgin_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return imgin_gray, imgin_color
    return None, None

# Chức năng chương 3
def process_chapter3(imgin_gray, imgin_color):
    chapter3_options = [
        "Negative", "Logarit", "Power", "PiecewiseLinear", 
        "Histogram", "HistEqual", "HistEqualColor", "LocalHist", 
        "HistStat", "BoxFilter", "LowpassGauss", "Threshold", 
        "MedianFilter", "Sharpen", "Gradient"
    ]
    
    chapter3_selected = st._main.selectbox("Select an option", chapter3_options)
    
    # Ánh xạ các lựa chọn với hàm tương ứng
    functions_map = {
        "Negative": lambda: c3.Negative(imgin_gray),
        "Logarit": lambda: c3.Logarit(imgin_gray),
        "Power": lambda: c3.Power(imgin_gray),
        "PiecewiseLinear": lambda: c3.PiecewiseLinear(imgin_gray),
        "Histogram": lambda: c3.Histogram(imgin_gray),
        "HistEqual": lambda: c3.HistEqual(imgin_gray),
        "HistEqualColor": lambda: c3.HistEqualColor(imgin_color),
        "LocalHist": lambda: c3.LocalHist(imgin_gray),
        "HistStat": lambda: c3.HistStat(imgin_gray),
        "BoxFilter": lambda: c3.BoxFilter(imgin_gray),
        "LowpassGauss": lambda: c3.MyBoxFilter(imgin_gray),
        "Threshold": lambda: c3.Threshold(imgin_gray),
        "MedianFilter": lambda: c3.MedianFilter(imgin_gray),
        "Sharpen": lambda: c3.Sharpen(imgin_gray),
        "Gradient": lambda: c3.Gradient(imgin_gray),
    }
    
    # Gọi hàm phù hợp dựa vào lựa chọn
    return functions_map[chapter3_selected]()

# Chức năng chương 4
def process_chapter4(imgin_gray):
    chapter4_options = ["Spectrum", "RemoveInterference", "RemoveMotionBlur", "DemotionWeiner", "RemoveMoire", "DemotionWeinerNoise"]
    
    chapter4_selected = st._main.selectbox("Select an option", chapter4_options)
    functions_map = {
        "Spectrum": lambda: c4.Spectrum(imgin_gray),
        "RemoveInterference": lambda: c4.RemoveInterference(imgin_gray),
        "RemoveMotionBlur": lambda: c4.RemoveMotionBlur(imgin_gray),
        "DemotionWeiner": lambda: c4.DemotionWeiner(imgin_gray),
        "RemoveMoire": lambda: c4.RemoveMoire(imgin_gray),
        "DemotionWeinerNoise": lambda: c4.DemotionWeinerNoise(imgin_gray),
    }
    return functions_map[chapter4_selected]()

# Chức năng chương 5
def process_chapter5(imgin_gray):
    chapter5_options = ["CreateMotionNoise", "DenoiseMotion", "DenoisestMotion"]
    
    chapter5_selected = st._main.selectbox("Select an option", chapter5_options)
    functions_map = {
        "CreateMotionNoise": lambda: c5.CreateMotionNoise(imgin_gray),
        "DenoiseMotion": lambda: c5.DenoiseMotion(imgin_gray),
        "DenoisestMotion": lambda: c5.DenoiseMotion(c5.DenoisestMotion(imgin_gray)),
    }
    return functions_map[chapter5_selected]()
    

# Chức năng chương 9
def process_chapter9(imgin_gray):
    chapter9_options = [
        "Erosion", "Dilation", "OpeningClosing", "Boundary", 
        "HoleFilling", "HoleFillingMouse", "ConnectedComponent", "CountRice"
    ]
    
    chapter9_selected = st._main.selectbox("Select an option", chapter9_options)
    
    functions_map = {
        "Erosion": lambda: c9.Erosion(imgin_gray),
        "Dilation": lambda: c9.Dilation(imgin_gray),
        "OpeningClosing": lambda: c9.OpeningClosing(imgin_gray),
        "Boundary": lambda: c9.Boundary(imgin_gray),
        "HoleFilling": lambda: c9.HoleFilling(imgin_gray),
        "HoleFillingMouse": lambda: c9.HoleFillingMouse(imgin_gray),
        "ConnectedComponent": lambda: c9.ConnectedComponent(imgin_gray),
        "CountRice": lambda: c9.CountRice(imgin_gray),
    }
    
    return functions_map[chapter9_selected]()

# Hàm chính
def main():
    setup_page()
    
    imgin_gray, imgin_color = upload_image()
    
    if imgin_gray is not None:
        # Tạo thanh bên chọn chương
        chapter_options = ["Chapter 3", "Chapter 4", "Chapter 5", "Chapter 9"]
        selected_chapter = st._main.selectbox("Select an option", chapter_options)
        
        # Xử lý ảnh theo chương được chọn
        if selected_chapter == "Chapter 3":
            processed_image = process_chapter3(imgin_gray, imgin_color)
        elif selected_chapter == "Chapter 4":
            processed_image = process_chapter4(imgin_gray)
        elif selected_chapter == "Chapter 5":
            processed_image = process_chapter5(imgin_gray)
        elif selected_chapter == "Chapter 9":
            processed_image = process_chapter9(imgin_gray)
            
        # Hiển thị ảnh gốc và ảnh đã xử lý
        st.subheader("Original Image and Processed Image")
        st.image([imgin_gray, processed_image], width=350)
    
    st.button("Run")

if __name__ == "__main__":
    main()