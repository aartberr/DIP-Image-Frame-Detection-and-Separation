#functions
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import math
from scipy.ndimage import convolve

########################################
#a_Hough_transform_demo.py

#reduce image detail for easier processing
def downsample_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    downsampled_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return downsampled_image

#perform Hough Transform on a binary image to detect lines and return the Hough space matrix, line parameters, and edge pixels
def my_hough_transform(binary_image, d_rho, d_theta,n):
    N1, N2 = binary_image.shape

    #theta range
    theta_max = np.pi  
    theta_range = np.arange(0, theta_max, d_theta)
    
    #rho range
    diag_len = int(np.sqrt(N1**2 + N2**2))
    rhos = np.arange(-diag_len, diag_len, d_rho)
    
    #accumulator array
    H = np.zeros((len(rhos), len(theta_range)), dtype=int)
    
    cos_t = np.cos(theta_range)
    sin_t = np.sin(theta_range)
    
    #indices of nonzero (edge) pixels
    y_idxs, x_idxs = np.nonzero(binary_image)
    
    #Hough Transform
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        rho_values = (x * cos_t + y * sin_t).astype(int)
        for j in range(len(theta_range)):
            rho = rho_values[j]
            rho_idx = np.where(rhos == rho)[0][0]
            H[rho_idx, j] += 1
    
    #find local maxima
    local_maxima = []
    window_size=15
    for i in range(0, H.shape[0]):
        for j in range(0, H.shape[1]):
            if i < window_size//2 or j < window_size//2:
                if j < window_size//2 and i < window_size//2:
                    window = H[0 : i + window_size//2 + 1, 0 : j + window_size//2 + 1]
                elif i >= window_size//2:
                    window = H[i - window_size//2 : i + window_size//2 + 1, 0 : j + window_size//2 + 1]
                else:
                    window = H[0 : i + window_size//2 + 1, j - window_size//2 : j + window_size//2 + 1]
            elif i > H.shape[0] - window_size//2 or j > H.shape[1] - window_size//2:
                if j > H.shape[1] - window_size//2 and i > H.shape[0] - window_size//2:
                    window = H[H.shape[0] - window_size//2 : H.shape[0], H.shape[1] - window_size//2 : H.shape[1]]
                elif i <= H.shape[0] - window_size//2:
                    window = H[i - window_size//2 : i + window_size//2 + 1, H.shape[1] - window_size//2 : H.shape[1]]
                else:
                    window = H[H.shape[0] - window_size//2 : H.shape[0], j - window_size//2 : j + window_size//2 + 1]
            else:
                window = H[i - window_size//2 : i + window_size//2 + 1, j - window_size//2 : j + window_size//2 + 1]
        
            if H[i, j] == np.max(window) and H[i, j] > 0:
                local_maxima.append((i, j, H[i, j]))


    # Sort and get top n maxima
    local_maxima = sorted(local_maxima, key=lambda x: x[2], reverse=True)[:n]

    # Create matrix L with the parameters rho and theta of the n most powerful lines
    L = np.array([(rhos[rho_idx], theta_range[theta_idx]) for rho_idx, theta_idx,value in local_maxima])

    # Mark points that belong to the n detected lines
    detected_points = np.zeros(binary_image.shape, dtype=bool)
    for rho, theta in L:
        for i in range(binary_image.shape[0]):
            for j in range(binary_image.shape[1]):
                if binary_image[i, j]:
                    rho_calc = int(i * np.cos(theta) + j * np.sin(theta))
                    if np.abs(rho_calc - rho) < d_rho:
                        detected_points[i, j] = True

    # Points that don't belong to the n detected lines
    res = binary_image & ~detected_points
    
    return H, L, res

#draw lines on an image based on Hough Transform parameters and scale them to match real image detail
def draw_detected_lines(image, lines, scale_factor):
    height, width = image.shape[:2]
    for rho, theta in lines:
        # Scale rho back to original size
        rho = rho / scale_factor
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))

        # Clip the line to the image boundaries
        x1, y1 = clip_line_to_bounds((height, width), x1, y1, x0, y0)
        x2, y2 = clip_line_to_bounds((height, width), x2, y2, x0, y0)
        
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image

 #clip a line defined by two points to fit within the boundaries of the given image shape
def clip_line_to_bounds(shape, x, y, x0, y0):
    height, width = shape[:2]
    if x < 0:
        y += (y0 - y) * (0 - x) / (x0 - x)
        x = 0
    elif x >= width:
        y += (y0 - y) * (width - 1 - x) / (x0 - x)
        x = width - 1
    if y < 0:
        x += (x0 - x) * (0 - y) / (y0 - y)
        y = 0
    elif y >= height:
        x += (x0 - x) * (height - 1 - y) / (y0 - y)
        y = height - 1
    return int(x), int(y)

############################################################ 
#b_Corner_detection.py

#apply Harris corner detection algorithm to an image and return the corner response values
def my_corner_harris(img, k, sigma):
    N1, N2 = img.shape
    padding = round(2 * sigma)

    #partial derivatives
    I1 = np.gradient(img, axis=0)  
    I2 = np.gradient(img, axis=1) 

    #pad I1 and I2 with zeros
    padded_I1 = np.pad(I1, ((padding, padding), (padding, padding)), mode='constant')
    padded_I2 = np.pad(I2, ((padding, padding), (padding, padding)), mode='constant')

    M11 = np.zeros((N1, N2))
    M12 = np.zeros((N1, N2))
    M22 = np.zeros((N1, N2))
    det_M = np.zeros((N1, N2))

    #compute M for each pixel
    for u1 in range(-padding, padding + 1):
        for u2 in range(-padding, padding + 1):
            w = np.exp(- (u1**2 + u2**2) / (2 * sigma**2))  

            shifted_I1 = padded_I1[padding + u1:N1 + padding + u1, padding + u2:N2 + padding + u2]
            shifted_I2 = padded_I2[padding + u1:N1 + padding + u1, padding + u2:N2 + padding + u2]

            M11 += w * (shifted_I1 ** 2)
            M12 += w * (shifted_I1 * shifted_I2)
            M22 += w * (shifted_I2 ** 2)

    det_M = M11 * M22 - M12 ** 2
    trace_M = M11 + M22

    harris_response = det_M - k * (trace_M ** 2)

    return harris_response

#find the local maxima in the Harris corner response matrix that exceed a given threshold and return their positions
def my_corner_peaks(harris_response, rel_threshold):
    absolute_threshold = harris_response.max() * rel_threshold
    thresholded_response = (harris_response > absolute_threshold)

    corner_locations = []
    for i in range(1, harris_response.shape[0] - 1):
        for j in range(1, harris_response.shape[1] - 1):
            if thresholded_response[i, j]:
                if harris_response[i, j] == np.max(harris_response[i-1:i+2, j-1:j+2]):
                    corner_locations.append((i, j))

    return np.array(corner_locations)

############################################################ 
#c_Image_rotation.py

#perform bilinear interpolation to estimate pixel values at non-integer coordinates on the image
def bilinear_interpolation(img, x, y):
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    
    if x0 < 0 or x1 >= img.shape[0] or y0 < 0 or y1 >= img.shape[1]:
        return 0  
    
    up = img[x0, y0-1]
    down = img[x0, y1]
    right = img[x1, y0]
    left = img[x0-1, y0]

    return 1/4* up + 1/4* down + 1/4* right + 1/4* left

 #rotate the image by a specified angle in radians, using interpolation to compute the pixel values in the rotated image
def my_img_rotation(img, angle):
    #angle = -angle  # reverse the angle to rotate counterclockwise
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    N1, N2 = img.shape[:2]

    #new image dimensions
    new_N1 = int(np.abs(N1 * cos_theta) + np.abs(N2 * sin_theta))
    new_N2 = int(np.abs(N1 * sin_theta) + np.abs(N2 * cos_theta))

    if len(img.shape) == 3: 
        rotated_img = np.zeros((new_N1, new_N2, img.shape[2]), dtype=img.dtype)
    else:  
        rotated_img = np.zeros((new_N1, new_N2), dtype=img.dtype)

    #find the center of the original and new images
    original_center = (N1 / 2, N2 / 2)
    new_center = (new_N1 / 2, new_N2 / 2)

    for i in range(new_N1):
        for j in range(new_N2):
            #set the center pixel in the new image to be (0,0) 
            x = i - new_center[0]
            y = j - new_center[1]

            #find the corresponding coordinates in the original image
            original_x = cos_theta * x + sin_theta * y
            original_y = -sin_theta * x + cos_theta * y

            #set the pixel (0,0) in the up left corner (as it is in the original image)
            original_x += original_center[0]
            original_y += original_center[1]

            if len(img.shape) == 3: 
                for c in range(img.shape[2]):
                    rotated_img[i, j, c] = bilinear_interpolation(img[:, :, c], original_x, original_y)
            else: 
                rotated_img[i, j] = bilinear_interpolation(img, original_x, original_y)

    return rotated_img

##################################
#d_Image_frame_extractor.py

#check if a given point is close to any element in an array within a specified threshold
def is_close(array, point, threshold):
    a, b = point
    for i in range(len(array)):
        close_x = np.abs(array[i, 0] - b) < threshold
        close_y = np.abs(array[i, 1] - a) < threshold
        if close_x and close_y:
            return 1
    return 0

#crop an image based on the specified corner coordinates
def crop_image(image, corners):
    x_coords = corners[::2]
    y_coords = corners[1::2]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    cropped_image = image.crop((min_x, min_y, max_x, max_y))

    return cropped_image
