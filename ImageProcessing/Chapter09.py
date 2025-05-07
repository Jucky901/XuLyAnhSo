import cv2
import numpy as np

L = 256

def Erosion(imgin):
    w = cv2. getStructuringElement(cv2.MORPH_RECT, (45,45))
    imgout = cv2.erode(imgin,w)
    return imgout

def Dilation(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    imgout = cv2.erode(imgin,w)
    return imgout
def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    temp = cv2.erode(imgin, w)
    return cv2.subtract(imgin, temp)

def Contour(imgin):
    img_color = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2)  # Màu xanh
    return img_color
