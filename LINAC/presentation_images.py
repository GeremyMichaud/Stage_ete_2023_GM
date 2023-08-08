import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt


def draw_rectangle(image, width=50, height=30, color=(0, 0, 255), thickness=2, y_offset=-100):
    image_height, image_width = image.shape[:2]
    mid_x, mid_y = image_width // 2, image_height // 2 + y_offset
    half_width = width // 2
    half_height = height // 2

    top_left = (mid_x - half_width, mid_y - half_height)
    bottom_right = (mid_x + half_width, mid_y + half_height)

    # Draw the rectangle on a copy of the image
    image_8bit = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    image_rgb = cv.cvtColor(image_8bit, cv.COLOR_GRAY2RGB)
    cv.rectangle(image_rgb, top_left, bottom_right, color, thickness)

    # Extract the region of interest (ROI) inside the rectangle
    roi = image[mid_y - half_height:mid_y + half_height, mid_x - half_width:mid_x + half_width]

    # Calculate the average grayscale value inside the ROI
    average_grayscale = np.mean(roi)

    return image_rgb, average_grayscale

if __name__ == "__main__":
    # Use glob to get all image paths
    image_paths = glob.glob('Measurements/2023-07-10/Raw_Data/18MV/unpol_15sec_*.png')

    # Load the images using OpenCV
    images = [cv.imread(path, cv.IMREAD_UNCHANGED) for path in image_paths]
    draw_image = cv.imread('Measurements/2023-07-10/Improved_Data/18MV/unpol.png', cv.IMREAD_UNCHANGED)

    if not images:
        print("Error: No images found or invalid image format.")
    else:
        # Draw a rectangle on the first image with an upward offset of 20 pixels
        image_with_rectangle, average_grayscale = draw_rectangle(draw_image)

        # Save the image with the rectangle
        cv.imwrite('Cerenkov/image_with_rectangle.png', image_with_rectangle)

        # Calculate the average grayscale value for all images
        grayscales = [draw_rectangle(image)[1] for image in images]
        mean = np.mean(grayscales)
        std = np.std(grayscales)
        print("Mean Gray Value: {0:.3e}".format(mean))
        print("Standard Deviation of Gray Values: {0:.3e}".format(std))
