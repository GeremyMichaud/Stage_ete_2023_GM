import cv2
from astropy.io import fits
from camera_calibrator import CameraCalibrator
import numpy as np
import glob


# Charger l'image FITS en niveaux de gris
fit_image = fits.open("Measurements/2023-06-27/Calibration/1sec_0001.fit")
image_data = fit_image[0].data
image_damier = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Dimensions du damier (nombre de carrés en largeur et en hauteur)
damier_width = 7
damier_height = 10



def fits_to_16bit_png(fits_path, png_path):
    # Load FITS image and extract data
    hdulist = fits.open(fits_path)
    data = hdulist[0].data

    # Normalize and scale to 16-bit range (0-65535)
    data = (data - data.min()) / (data.max() - data.min()) * 65535
    data = data.astype('uint16')

    # Save as 16-bit PNG using OpenCV
    cv2.imwrite(png_path, data)

    hdulist.close()

# Replace 'input.fits' and 'output.png' with your actual file paths
input_fits_path = 'Measurements/2023-06-27/Calibration/1sec_0001.fit'
output_png_path = 'Measurements/2023-06-27/test.png'

fits_to_16bit_png(input_fits_path, output_png_path)