import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tifffile import imread

GT_SHAPE = (1,  128, 128)
S1_SHAPE = (4,  128, 128)
S2_SHAPE = (5, 128, 128)
IMG_SIZE = (128, 128)

s2_min = [280.00, 350.00, 280.00, 970.00, 400.00]
s2_max = [2150.00, 2300.00, 2500.00, 5000.00, 2300.00]
s1_min = [-17, -24, -17, -25]
s1_max = [-3, -10, -4, -10]

min_label = 0.0
max_label = 60.0 #adjust as needed

def scale_img(matrix):
    min_values = np.array(s1_min)
    max_values = np.array(s1_max)

    # Reshape matrix
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float32)

    # Scale by min/max
    matrix = np.nan_to_num((matrix - min_values[None, :]) / (max_values[None, :] - min_values[None, :] + 0.1))
    matrix = np.reshape(matrix, [w, h, d])
    return matrix.clip(0, 1)
    
def GRD_toRGB_S1(path_S1):
    # Read VV/VH bands
    sar_img = imread(path_S1)    
    vv_ASC = sar_img[:, :, 0]
    vh_ASC = sar_img[:, :, 1]
    vv_DSC = sar_img[:, :, 2]
    vh_DSC = sar_img[:, :, 3]
    vv_ASC = cv2.resize(vv_ASC, (IMG_SIZE[0], IMG_SIZE[1]), interpolation = cv2.INTER_AREA)
    vh_ASC = cv2.resize(vh_ASC, (IMG_SIZE[0], IMG_SIZE[1]), interpolation = cv2.INTER_AREA)
    vv_DSC = cv2.resize(vv_DSC, (IMG_SIZE[0], IMG_SIZE[1]), interpolation = cv2.INTER_AREA)
    vh_DSC = cv2.resize(vh_DSC, (IMG_SIZE[0], IMG_SIZE[1]), interpolation = cv2.INTER_AREA)
    x_img = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 4), dtype=np.float32)
    x_img[:, :, 0] = vv_ASC
    x_img[:, :, 1] = vh_ASC
    x_img[:, :, 2] = vv_DSC
    x_img[:, :, 3] = vh_DSC
    return scale_img(x_img)

def scale_imgS2(matrix):    
    min_values = np.array(s2_min)
    max_values = np.array(s2_max)
    # Reshape matrix
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float32)
    # Scale by min/max
    matrix = np.nan_to_num((matrix - min_values[None, :]) / (max_values[None, :] - min_values[None, :] + 0.1))
    matrix = np.reshape(matrix, [w, h, d])
    return matrix.clip(0, 1)

def GRD_toRGB_S2(path_S2):
    # B , g, r, NIR, SWIR
    s2_img = imread(path_S2)    
    x_img = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 5), dtype=np.float32)
    x_img[:, :, 0] = cv2.resize(s2_img[:, :, 0], (IMG_SIZE[0], IMG_SIZE[1]), interpolation = cv2.INTER_AREA)
    x_img[:, :, 1] = cv2.resize(s2_img[:, :, 1], (IMG_SIZE[0], IMG_SIZE[1]), interpolation = cv2.INTER_AREA)
    x_img[:, :, 2] = cv2.resize(s2_img[:, :, 2], (IMG_SIZE[0], IMG_SIZE[1]), interpolation = cv2.INTER_AREA)
    x_img[:, :, 3] = cv2.resize(s2_img[:, :, 6], (IMG_SIZE[0], IMG_SIZE[1]), interpolation = cv2.INTER_AREA)
    x_img[:, :, 4] = cv2.resize(s2_img[:, :, 9], (IMG_SIZE[0], IMG_SIZE[1]), interpolation = cv2.INTER_AREA)
    x_img =  scale_imgS2(x_img)
    return x_img

def read_raster(data_path, return_zeros=False, data_shape=None):
    if os.path.isfile(data_path):
        raster = rasterio.open(data_path)
        data = raster.read()
    else:
        if return_zeros:
            assert data_shape is not None
            data = np.zeros(data_shape).astype(np.float32)
        else:
            data = None
    return data

def normalize(data):    
    data = (data - min_label) / (max_label-min_label)
    return data.clip(0, 1)

def recover_label(data):
    return (data * (max_label-min_label)) +  min_label