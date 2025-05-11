import argparse
import logging
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# from utils.data_loading import BasicDataset 

import sys
# Lấy đường dẫn của thư mục cha của thư mục chứa script hiện tại (XuLyAnhSo)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
print(sys.path)
print(parent_dir)
from model.unet import UNet
net = UNet(n_channels=3, n_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device=device)
net.load_state_dict(torch.load("model/lanedetectionSegmentation.pth"))
scale_factor = 0.5
out_threshold = 0.5

def preprocess(pil_img, scale, is_mask):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img = np.asarray(pil_img)

    if is_mask:
        if img.ndim == 3:  # Nếu mask là RGB, chuyển sang grayscale
            img = img[:, :, 0]
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        mask[img == 1] = 1  # Đảm bảo mask chỉ có giá trị 0 và 1
        return mask
    else:
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))
        if (img > 1).any():
            img = img / 255.0
        return img

def predict_img(full_img):
    net.eval()
    img = torch.from_numpy(preprocess( full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()



import streamlit as st

# Cấu hình trang Streamlit
def setup_page():  
    st.title("Lane Segmentation")
    st.write("Dự đoán phân đoạn làn đường từ ảnh đầu vào")


# Tải lên ảnh
def upload_image():
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg', 'tif'])
    
    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        imgin_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE) 
        imgin_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return imgin_gray, imgin_color
    return None, None
def main():
    # Thiết lập trang
    setup_page()

    # Tải lên ảnh
    uploaded_file = st.file_uploader("Chọn ảnh đầu vào", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Đọc ảnh từ file tải lên
        img = Image.open(uploaded_file)
        st.image(img, caption='Ảnh đầu vào', use_column_width=True)

        # Dự đoán phân đoạn
        mask = predict_img(img)
        mask = mask.astype(np.uint8) * 255

        # Hiển thị ảnh phân đoạn
        st.image(mask, caption='Ảnh phân đoạn', use_column_width=True)
if __name__ == "__main__":
    # Chạy ứng dụng Streamlit
    main()
# st.title("Lane Segmentation")
# st.write("Dự đoán phân đoạn làn đường từ ảnh đầu vào")
# uploaded_file = st.file_uploader("Chọn ảnh đầu vào", type=["jpg", "jpeg", "png"])
# if uploaded_file is not None:
#     # Đọc ảnh từ file tải lên
#     img = Image.open(uploaded_file)
#     st.image(img, caption='Ảnh đầu vào', use_column_width=True)

#     # Dự đoán phân đoạn
#     mask = predict_img(img)
#     mask = mask.astype(np.uint8) * 255

#     # Hiển thị ảnh phân đoạn
#     st.image(mask, caption='Ảnh phân đoạn', use_column_width=True)

# img = cv2.imread("pictures/lanesegmentation/lane_02209.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = Image.fromarray(img)
# mask = predict_img(img)
# mask = mask.astype(np.uint8) * 255
# cv2.imshow("Mask", mask)
# cv2.imshow("Original Image", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()