import numpy as np
import cv2

L = 256
def Spectrum(imgin):

    f = imgin.astype(np.float32)/(L-1)
    F = np.fft.fft2(f)


    F = np.fft.fftshift(F)
    FR = F.real.copy()
    FI = F.imag.copy()
    S = np.sqrt(FR**2 + FI**2)
    # S = np.log(1 + S)  # Logarithmic scaling for better visibility
    S = np.clip(S, 0, L-1)
    S = S.astype(np.uint8)
    imgout = S.copy()
    return imgout
#1

def CreateNotchFilter(M, N):
    H = np.ones((M, N), np.complex64)
    H.imag = 0.0

    u1, v1 = 44, 58
    u2, v2 = 86, 58
    u3, v3 = 40, 119
    u4, v4 = 82, 119

    u5, v5 = M - u1, N - v1
    u6, v6 = M - u2, N - v2
    u7, v7 = M - u3, N - v3
    u8, v8 = M - u4, N - v4
    D0 = 10
    for u in range(M):
        for v in range(N):
            d1 = np.sqrt((u - u1) ** 2 + (v - v1) ** 2)
            d2 = np.sqrt((u - u2) ** 2 + (v - v2) ** 2)
            d3 = np.sqrt((u - u3) ** 2 + (v - v3) ** 2)
            d4 = np.sqrt((u - u4) ** 2 + (v - v4) ** 2)
            d5 = np.sqrt((u - u5) ** 2 + (v - v5) ** 2)
            d6 = np.sqrt((u - u6) ** 2 + (v - v6) ** 2)
            d7 = np.sqrt((u - u7) ** 2 + (v - v7) ** 2)
            d8 = np.sqrt((u - u8) ** 2 + (v - v8) ** 2)

            if d1 < D0 or d5 < D0 or d3 < D0 or d7 < D0 or d2 < D0 or d6 < D0 or d4 < D0 or d8 < D0:
                H.real[u,v] = 0
    return H
#2
def RemoveMoire(imgin):
    M, N = imgin.shape
    H = CreateNotchFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout
#3
def FrequencyFilter(imgin, H):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = imgin

    for x in range(M):
        for y in range(N):
            fp[x, y] *= (-1) ** (x + y)

    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
    G = cv2.mulSpectrums(F, H, flags=0)

    g = cv2.idft(G, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    g = g[:M, :N]

    for x in range(M):
        for y in range(N):
            g[x, y] *= (-1) ** (x + y)

    g = np.clip(g, 0, L - 1)
    imgout = g.astype(np.uint8)
    return imgout
#4
def FrequencyFiltering(imgin, H):
    f = imgin.astype(np.float32)

    # Step 1: DFT
    F = np.fft.fft2(f)

    # Step 2: Shift to the center of the image
    F = np.fft.fftshift(F)

    # Step 3: Multiply F with H (use only the first channel of H)
    G = F * H.real
    
    # Step 4: Shift back
    G = np.fft.ifftshift(G)

    # Step 5: IDFT
    g = np.fft.ifft2(G)
    gR = g.real.copy()
    gR = np.clip(gR, 0, L-1)
    imgout = gR.astype(np.uint8)
    return imgout

def CreateInterferenceFilter(imgin):
    M,N  = imgin.shape
    H = np.ones((M,N), np.complex64)
    H.imag=0.0
    D0 = 7
    D1 = 7
    for u in range(0,M):
        for v in range(0,N):
            if u not in range(M//2-D0,M//2+D0+1):
                if v in range(N//2-D1, N//2+D1+1):
                    H.real[u,v]= 0
    return H
def RemoveInterference(imgin):
    M, N = imgin.shape
    H = CreateInterferenceFilter(imgin)
    imgout = FrequencyFiltering(imgin,H)
    return imgout

def CreateMotionFilter(M,N):
    H = np.zeros((M,N), np.complex64)
    a = 0.1
    b = 0.1
    T = 1.0
    # for u in range(M):
    #     for v in range(N):
    #         phi = np.pi * (a * (u-M//2) + b * (v-N//2))
    #         if abs(phi) < 1e-6:
    #             RE  = T
    #             IM = 0.0
    #         else:
    #             RE = T*np.sin(phi)/phi * np.cos(phi)
    #             IM = T*np.sin(phi)/phi * np.sin(phi)
    #         H.real[u,v] = RE
    #         H.imag[u,v] = IM
    for u in range(M):
        for v in range(N):
            phi = np.pi * (a * (u-M//2) + b * (v-N//2))
            if abs(phi) < 1e-6:
                phi = phi_prev
            
            RE = T*np.sin(phi)/phi * np.cos(phi)
            IM = T*np.sin(phi)/phi * np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def RemoveMotionBlur(imgin):
    M, N = imgin.shape
    H = CreateMotionFilter(M,N)
    imgout = FrequencyFiltering(imgin,H)
    return imgout

# def CreateMotionFilter(M,N):
#     H = np.zeros((M,N), np.complex64)
#     a = 0.1
#     b = 0.1
#     T = 1.0
    # for u in range(M):
    #     for v in range(N):
    #         phi = np.pi * (a * (u-M//2) + b * (v-N//2))
    #         if abs(phi) < 1e-6:
    #             RE  = T
    #             IM = 0.0
    #         else:
    #             RE = T*np.sin(phi)/phi * np.cos(phi)
    #             IM = T*np.sin(phi)/phi * np.sin(phi)
    #         H.real[u,v] = RE
    #         H.imag[u,v] = IM
def CreateDeMotionFilter(M,N):
    H = np.zeros((M,N), np.complex64)
    a = 0.1
    b = 0.1
    T = 1.0
    phi_prev = 0.0
    for u in range(M):
        for v in range(N):
            phi = np.pi * (a * (u-M//2) + b * (v-N//2))
            temp = np.sin(phi)
            if abs(temp) < 1e-6:
                phi = phi_prev
            
            RE = phi/(T*np.sin(phi)) * np.cos(phi)
            IM = phi/T
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def RemoveMotionBlur(imgin):
    M, N = imgin.shape
    H = CreateDeMotionFilter(M,N)
    imgout = FrequencyFiltering(imgin,H)
    return imgout


def Create(M,N):
    H = np.zeros((M,N), np.complex64)
    a = 0.1
    b = 0.1
    T = 1.0
    phi_prev = 0.0
    for u in range(M):
        for v in range(N):
            phi = np.pi * (a * (u-M//2) + b * (v-N//2))
            temp = np.sin(phi)
            if abs(temp) < 1e-6:
                phi = phi_prev
            
            RE = phi/(T*np.sin(phi)) * np.cos(phi)
            IM = phi/T
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    P = H.real**2 + H.imag**2
    K = 1.0
    H = H*P/(P+K)
    return H
def DemotionWeiner(imgin): 
    M, N = imgin.shape
    H = Create(M,N)
    imgout = FrequencyFiltering(imgin,H)
    return imgout

def DemotionWeinerNoise(imgin):
    imgin = cv2.medianBlur(imgin,7)
    imgout = DemotionWeiner(imgin)
    return imgout
