from PIL import Image
import cv2 as cv
import numpy as np

# Load the grayscale 16-bit background image
image_background = cv.imread("Measurements/2023-07-24/Cube/cube.png", cv.IMREAD_UNCHANGED)

# Load the transparent PNG overlay image
image_overlay = cv.imread("Measurements/2023-07-24/Cube/image_transparente.png", cv.IMREAD_UNCHANGED)

# Resize the overlay image to match the background image if needed
if image_overlay.shape[:2] != image_background.shape[:2]:
    image_overlay = cv.resize(image_overlay, (image_background.shape[1], image_background.shape[0]))

# Extract the alpha channel from the overlay image
overlay_alpha = image_overlay[:, :, 3]

# Manually adjust contrast and brightness
alpha = 1.5
beta = 30

# Apply contrast and brightness adjustments to the RGB channels of the overlay image
overlay_rgb = image_overlay[:, :, :3]
adjusted_overlay_rgb = cv.convertScaleAbs(overlay_rgb, alpha=alpha, beta=beta)

# Create a new grayscale image with the same dimensions as the background
combined_image = np.copy(image_background)

# Apply the overlay using the alpha channel
for y in range(image_background.shape[0]):
    for x in range(image_background.shape[1]):
        alpha = overlay_alpha[y, x] / 255.0
        combined_image[y, x] = int((1 - alpha) * combined_image[y, x] + alpha * adjusted_overlay_rgb[y, x, 0])

# Show the final combined grayscale image
cv.imshow("Combined Image", combined_image)
cv.waitKey(0)
cv.destroyAllWindows()
