import cv2 as cv
import numpy as np
import glob
from matplotlib import cm


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

def process_and_combine_images():
    # Load the grayscale image
    image_beam = cv.imread("Measurements/2023-07-24/Improved_Data/6MV/3.png", cv.IMREAD_UNCHANGED)

    # Define the coordinates of the rectangle centered (x, y, width, height)
    center_x = image_beam.shape[1] // 2
    center_y = image_beam.shape[0] // 2
    rect_width = 200
    rect_height = 770

    # Create a mask for pixels inside the centered rectangle and above the threshold
    threshold = 3000
    mask = np.zeros_like(image_beam, dtype=np.uint8)
    mask[center_y - rect_height // 2:center_y + rect_height // 2, center_x - rect_width // 2:center_x + rect_width // 2] = np.where(image_beam[center_y - rect_height // 2:center_y + rect_height // 2,
        center_x - rect_width // 2:center_x + rect_width // 2] > threshold, 255, 0).astype(np.uint8)

    # Apply colormap (Inferno) to the masked region
    colormap_inferno = (cm.inferno(mask)[:, :, :3] * 255).astype(np.uint8)

    # Create a mask for the region outside the rectangle
    inverse_mask = cv.bitwise_not(mask)

    # Create a copy of the original image
    colored_image = cv.cvtColor(image_beam, cv.COLOR_GRAY2BGR)

    # Set the region outside the rectangle to zero
    colored_image[np.where(inverse_mask == 255)] = 0

    # Normalize to uint8 and blend the colormap with the grayscale image
    colored_image_blend = colored_image.astype(float) * (1 - colormap_inferno[..., 0:3] / 255.0)
    colored_image_blend += colormap_inferno[..., 0:3]

    # Set transparent pixels (black) to alpha value 0
    alpha_channel = (colored_image_blend[:, :, 0] != 0).astype(np.uint8) * 255
    colored_image_uint8 = np.dstack((colored_image_blend.astype(np.uint8), alpha_channel))

    # Load the grayscale 16-bit background image
    image_background = cv.imread("Measurements/2023-07-24/Cube/cube.png", cv.IMREAD_GRAYSCALE)

    # Create a new color image with the same dimensions as the background
    combined_image = cv.cvtColor(image_background, cv.COLOR_GRAY2BGR)

    # Manually adjust contrast and brightness for the overlay image
    alpha = 1.5
    beta = 30
    adjusted_overlay = cv.convertScaleAbs(colored_image_uint8[:, :, :3], alpha=alpha, beta=beta)
    adjusted_alpha_channel = colored_image_uint8[:, :, 3]

    # Apply the overlay using the alpha channel and adjusted overlay
    for y in range(image_background.shape[0]):
        for x in range(image_background.shape[1]):
            alpha = adjusted_alpha_channel[y, x] / 255.0
            combined_image[y, x] = (1 - alpha) * combined_image[y, x] + alpha * adjusted_overlay[y, x]

    # Save the final combined color image
    cv.imwrite("Measurements/2023-07-24/Cube/combined_image.png", combined_image)

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

    process_and_combine_images()
