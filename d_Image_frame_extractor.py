#my_lazy_scanner.py
import numpy as np
from PIL import Image
from functions import *
import cv2
import os
import matplotlib.pyplot as plt

# Get the grayscale image
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

# Perform Hough Transform
H, L, res = my_hough_transform(binary_image,1,np.pi/180,15)

# Draw detected lines on the original image
img_array = np.array(img)
img_with_lines = draw_detected_lines(img_array, L, scale_factor)

parallel_lines=[]
angle_tolerance = 0.02
for i in range(len(L)-1):
    for j in range(i+1,len(L)):
        if abs(L[i, 1] - L[j, 1]) < angle_tolerance and L[i, 0] != L[j, 0]:
            parallel_lines.append([L[i, 0], L[j, 0], L[i, 1]])

parallel = np.unique(parallel_lines, axis=0)

#find pairs of angles of vertical lines
theta_values= np.unique(parallel[:,2], axis=0)
tuples_of_vertical_lines=[]
angle_tolerance=0.02
for i in range(len(theta_values)-1):
    for j in range(i+1,len(theta_values)):
        if abs(theta_values[j] - theta_values[i] - np.pi / 2) < angle_tolerance:
            tuples_of_vertical_lines.append([theta_values[i],theta_values[j]])

#if an angle matches with more than one angle choose the one not in other couple or the first
tuple_to_remove=[]
for k in range(2):
    for i in range(len(tuples_of_vertical_lines)-1):
        if tuples_of_vertical_lines[i][k] == tuples_of_vertical_lines[i+1][k]:
            element_to_check = tuples_of_vertical_lines[i][abs(k-1)]
            element_found = any(element_to_check in t for t in tuples_of_vertical_lines)
            if element_found:
                parallel[np.where(parallel==element_to_check)[0],2] = tuples_of_vertical_lines[i+1][abs(k-1)] 
                tuple_to_remove.append(tuples_of_vertical_lines[i])

for tpl in tuple_to_remove:
    if tpl in tuples_of_vertical_lines:
        tuples_of_vertical_lines.remove(tpl)

possible_frames = []
for tuple in tuples_of_vertical_lines:
    #find indices  of the tuple on array parallel that contain the same element with the first element of the tuple
    #the same for the second
    indices1 = [index for index, value in enumerate(parallel[:,2]) if value == tuple[0]]
    indices2 = [index for index, value in enumerate(parallel[:,2]) if value == tuple[1]]
    for index1 in indices1:
        for index2 in indices2:
            possible_frames.append(np.concatenate((parallel[index1,:], parallel[index2,:])))

#find intersection points
intersection_points= []
for frame in possible_frames:
    rho1, rho2, theta1 = frame[0], frame[1], frame[2]
    rho3, rho4, theta2 = frame[3], frame[4], frame[5]
    
    if theta1 % np.pi == 0:  
        x0 = rho1
        y0 = rho3
        x1 = x0
        y1 = rho4
        x2 = rho2
        y2 = rho3
        x3 = x2
        y3 = rho4
    elif theta1 % np.pi == np.pi/2: 
        x0 = rho3
        y0 = rho1
        x1 = rho4
        y1 = y0
        x2 = rho3
        y2 = rho2
        x3 = rho4
        y3 = y2
    else:  
        lamda1 = -1 / np.tan(theta1)
        lamda2 = -1 / np.tan(theta2)
        x0 = ( rho3 * np.sin(theta2) - rho1 * np.sin(theta1) + lamda1 * rho1 * np.cos(theta1) - lamda2 * rho3 * np.cos(theta2) ) / (lamda1 - lamda2)
        y0 = lamda1 * (x0 - rho1 * np.cos(theta1)) + rho1* np.sin(theta1)
        x1 = ( rho4 * np.sin(theta2) - rho1 * np.sin(theta1) + lamda1 * rho1 * np.cos(theta1) - lamda2 * rho4 * np.cos(theta2) ) / (lamda1 - lamda2)
        y1 = lamda1 * (x1 - rho1* np.cos(theta1)) + rho1 * np.sin(theta1)
        x2 = ( rho3 * np.sin(theta2) - rho2 * np.sin(theta1) + lamda1 * rho2 * np.cos(theta1) - lamda2 * rho3 * np.cos(theta2) ) / (lamda1 - lamda2)
        y2 = lamda1 * (x2 - rho2 * np.cos(theta1)) + rho2 * np.sin(theta1)
        x3 = ( rho4 * np.sin(theta2) - rho2 * np.sin(theta1) + lamda1 * rho2 * np.cos(theta1) - lamda2 * rho4 * np.cos(theta2) ) / (lamda1 - lamda2)
        y3 = lamda1 * (x3 - rho2 * np.cos(theta1)) + rho2 * np.sin(theta1)
    intersection_points.append([x0,y0,x1,y1,x2,y2,x3,y3])

intersection_points= np.array(intersection_points) / scale_factor

#apply harris corner detection
normalized_image= binary_image / np.max(binary_image)

k = 0.05
sigma = 2.5
harris_response = my_corner_harris(normalized_image, k, sigma)

rel_threshold = 0.1
corner_locations = my_corner_peaks(harris_response, rel_threshold) / scale_factor

#for each possible frame we ckeck if it is actually a frame
threshold=100
rows_of_frames=[]
for row in range(len(intersection_points)):
    real_corner_points=0
    for corner in range(4):
        possible_corner = intersection_points[row, 2*corner], intersection_points[row, 2*corner + 1]
        real_corner_points += is_close(corner_locations, possible_corner, threshold)
    #check if all corners are close to corner peak, if yes cosider it a frame
    if real_corner_points == 4:
        rows_of_frames.append(row)

#check for any uneccessary frames
row_frame_to_remove=[]
for row1 in rows_of_frames:
    for row2 in rows_of_frames:
        x_coords1 = intersection_points[row1, ::2]  #take columns 0, 2, 4, 6
        y_coords1 = intersection_points[row1, 1::2]  #take columns 1, 3, 5, 7
        min_x1, max_x1 = min(x_coords1), max(x_coords1)
        min_y1, max_y1 = min(y_coords1), max(y_coords1)
        x_coords2 = intersection_points[row2, ::2]  #take columns 0, 2, 4, 6
        y_coords2 = intersection_points[row2, 1::2]  #take columns 1, 3, 5, 7
        min_x2, max_x2 = min(x_coords2), max(x_coords2)
        min_y2, max_y2 = min(y_coords2), max(y_coords2)
        #check if frame[row2] is inside frame[row1]
        if row1 != row2: 
            if min_x1 <= min_x2 and max_x2 <= max_x1 and min_y1 <= min_y2 and max_y2 <= max_y1 and row1 != row2:
                #count corners inside the frames
                corner_count1 = 0
                corner_count2 = 0
                for corner in corner_locations:
                    y, x = corner
                    if min_x1 <= x <= max_x1 and min_y1 <= y <= max_y1:
                        corner_count1 += 1
                        y, x = corner
                    if min_x2 <= x <= max_x2 and min_y2 <= y <= max_y2:
                        corner_count2 += 1
                #choose the one with the highest percentage of corners
                if corner_count2 >= corner_count1 * 0.5:
                    row_frame_to_remove.append(row1)
                else:
                    row_frame_to_remove.append(row2)

for row in np.unique(np.array(row_frame_to_remove)):
    rows_of_frames.remove(row)

file_name_without_extension = os.path.splitext(filename)[0]
for i,row in enumerate(rows_of_frames):
    cropped_image =crop_image(img, intersection_points[row, :])
    new_filename = f'{file_name_without_extension}_{i+1}.jpg'
    cropped_image.save(new_filename)