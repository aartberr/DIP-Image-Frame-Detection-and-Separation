#deliverable_1
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

#Hough Transform
H, L, res = my_hough_transform(binary_image,1,np.pi/180,15)


#plot Hough transform array and L points
ycoords = L[:,0] 
xcoords = L[:, 1] * 180 / np.pi
plt.figure(figsize=(6, 6))  
plt.imshow(H, aspect='auto', cmap='gray', extent=[0, 180, -H.shape[0]/2, H.shape[0]/2])
plt.colorbar(label='votes')
plt.scatter(xcoords.tolist(),ycoords.tolist(), color='red', s=4) 
plt.xlabel('theta')
plt.ylabel('rho')
plt.title('Hough Transform array')
plt.show()

#plot the detected lines on the original image
img_array = np.array(img)
img_with_lines = draw_detected_lines(img_array, L, scale_factor)

plt.figure(figsize=(6, 6))
plt.imshow(img_with_lines)
plt.title('Image with Detected Lines')
plt.axis('off')
plt.show()

