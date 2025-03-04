# Example usage
import numpy as np
from PIL import Image
from functions import *
import cv2
import matplotlib.pyplot as plt

# Set the filepath to the image file
filename = "input_img.jpg"

# Read the image into a PIL entity
img = Image.open(fp=filename)

# Keep only the Luminance component of the image
bw_img = img.convert("L")

# Convert the PIL image to a numpy array
bw_img_np = np.array(bw_img)

#downsample the image
scale_factor = 0.08
bw_img_np_downsampled = downsample_image(bw_img_np, scale_factor)

#Canny edge detection
low_threshold = 225
high_threshold = 275
binary_image = cv2.Canny(bw_img_np_downsampled, low_threshold, high_threshold)

normalized_image= binary_image / np.max(binary_image)

#Harris corner detection
k = 0.05
sigma = 2.5
harris_response = my_corner_harris(normalized_image, k, sigma)

#find corner locations
rel_threshold = 0.1
corner_locations = my_corner_peaks(harris_response, rel_threshold) / scale_factor

#plot the detected corners
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.scatter(corner_locations[:, 1], corner_locations[:, 0], color='red', s=8)
plt.title('Detected Corners')
plt.axis('off')
plt.show()


