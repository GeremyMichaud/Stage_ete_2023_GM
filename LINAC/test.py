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