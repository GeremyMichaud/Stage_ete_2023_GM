#!/usr/bin/env python

# Le module cv2 de la librairie OpenCV-Python est conçue pour résoudre les problèmes de vision par ordinateur
import cv2
# La librairie numpy contient la majorité des éléments mathématiques utilisés dans ce travail
import numpy as np
# Le module glob trouve tous les noms de chemin correspondant à un modèle spécifié
import glob
from astropy.io import fits

# Définir du nombre de coins intérieurs par ligne et colonne d'échiquier (points_per_row, points_per_colum)
CHECKERBOARD = (7,10)
# Spécifier les paramètres pour les critères d'arrêt
# Dans ce cas, 30 indique le nombre maximal d'itérations autorisées et 0.001 indique la précision (epsilon) à atteindre
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Création d'un vecteur pour stocker des vecteurs de points 3D pour chaque image en damier
objpoints = []
# Création d'un vecteur pour stocker des vecteurs de points 2D pour chaque image en damier
imgpoints = [] 


# Définir les coordonnées réelles pour les points 3D
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extraction du chemin d'images individuelles stockées dans un répertoire donné
#date = input("Enter the date of your data acquision (YYYY-MM-DD):")
date = "2023-06-27"
images = glob.glob(f"Measurements/{date}/Calibration/*")


for fits_path in images:
    # Lire le fichier FITS
    fits_data = fits.open(fits_path)
    fits_image_data = fits_data[0].data
    # Normaliser les données d'image si nécessaire
    # (convertir en plage 0-255 pour 8-bit JPEG)
    jpeg_image_data = cv2.normalize(fits_image_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Convertir les données d'image au format BGR
    gray = cv2.cvtColor(jpeg_image_data, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(gray, CHECKERBOARD, corners2, ret)

    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h,w = jpeg_image_data.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)