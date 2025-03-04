#deliverable_3.py
# Example usage
import numpy as np
from PIL import Image
from functions import *
import matplotlib.pyplot as plt

# Set the filepath to the image file
filename = "input_img.jpg"

# Read the image into a PIL entity
img = Image.open(fp=filename)

# Convert the PIL image to a numpy array
img_np = np.array(img)

#downsample the image
scale_factor = 0.25
img_np_downsampled = downsample_image(img_np, scale_factor)


angles = [54 * np.pi /180, 213 * np.pi / 180]
for angle in angles:
    rotated_img = my_img_rotation(img_np_downsampled, angle)
    plt.title('Rotated Image')
    plt.imshow(rotated_img)
    plt.show()