# Processing Algorithms for Image Seperation

This project was created as part of a course assignment for the **DIP (Digital Image Processing)** class at **ECE, AUTH University.**

This project implements various image processing algorithms to detect and separate different images within a larger image. The following algorithms are used:

1. **Hough Transform** for detecting the most prominent edges in an image.
2. **Harris Corner Detector** to find the corners in an image.
3. **Image Rotation** for rotating images by a specified angle.

By combining these methods, the scripts can identify edges, corners, and structured regions, allowing for image separation.

## Files

- **example_img.jpg**: An example image that contains two distinct images within a single image.
- **a_Hough_transform_demo.py**: Demonstrates the application of the Hough Transform on `input_img.jpg` to detect the strongest lines. It plots the Hough transform array and the image with blue lines highlighting the detected edges.
- **b_Corner_detection.py**: Applies the Harris Corner Detection on `input_img.jpg` and highlights the corners of the frame, potentially identifying the frame's corners.
- **c_Image_rotation.py**: Rotates `input_img.jpg` counterclockwise by a specified angle.
- **d_Image_frame_extractor.py**: Integrates all steps to detect lines, find parallel pairs, identify vertical parallel pairs (possible separators), and compute angle coordinates from corner detection. It filters out unnecessary regions and saves the cropped images. Rotation is not yet applied but may be used in future improvements.
- **functions.py**: Contains helper functions used across all scripts.

## Usage

Ensure your image file is named `input_img.jpg` and is placed in the same directory as the scripts..

Run any of the scripts:
```bash
python a_Hough_transform_demo.py
python b_Corner_detection.py
python c_Image_rotation.py
python d_Image_extractor.py
```
## Future Work

- Implement rotation correction before cropping to properly handle rotated images.
- Improve the selection criteria for choosing the most relevant detected regions.
