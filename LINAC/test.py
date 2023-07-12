import cv2
from astropy.io import fits
import glob
from camera_calibrator import CameraCalibrator

# Load the .fit image
fit_image = fits.open("Measurements/2023-06-27/Calibration/1sec_0001.fit")

# Extract the image data from the FITS file
image_data = fit_image[0].data

# Normalize the pixel values to the 0-255 range
image_damier = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

calibrator = CameraCalibrator((7, 10), glob.glob("Measurements/2023-06-27/Calibration/*"))

_, mtx, dist, _, _ = calibrator.calibrate_camera()

framesize = image_damier.shape[:2]
# Obtenir une nouvelle matrice de caméra optimale et une région d'intérêt (ROI) pour l'image corrigée
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, framesize[::-1], 1,  framesize[::-1])
# Appliquer la correction de distorsion à l'image
undistort = cv2.undistort(image_damier, mtx, dist, None, newCameraMatrix)
# Recadrer l'image à la région d'intérêt (ROI)
x, y, w, h = roi
improved_damier = undistort[y:y+h, x:x+w]

damier_width = 7
damier_height = 10

# Trouver les coins du damier dans l'image
_, corners = cv2.findChessboardCorners(improved_damier, (damier_width, damier_height))

# Sélectionner un carré spécifique du damier (ici, le carré au centre)
target_square_index = damier_width * (damier_height // 2) + (damier_width // 2)
target_square_corner = corners[target_square_index]

# Coordonnées du coin supérieur gauche du carré cible
x, y = target_square_corner.ravel()

# Dimensions du carré en pixels
square_size_pixels = 50  # Remplacez par la taille réelle du carré en pixels

# Calcul des coordonnées par rapport à l'axe central
central_axis_x = damier_width // 2
central_axis_y = damier_height // 2
relative_x = x - (central_axis_x * square_size_pixels)
relative_y = (central_axis_y * square_size_pixels) - y

# Affichage des résultats
print("Coordonnées du coin supérieur gauche du carré cible :")
print("X :", x)
print("Y :", y)
print("Coordonnées relatives par rapport à l'axe central :")
print("Relative X :", relative_x)
print("Relative Y :", relative_y)

# Dessiner un point sur le coin supérieur gauche du carré cible
cv2.circle(improved_damier[0], (x, y), 5, (0, 0, 255), -1)

# Dessiner un point sur les coordonnées relatives par rapport à l'axe central
relative_point_x = int((central_axis_x * square_size_pixels) + relative_x)
relative_point_y = int((central_axis_y * square_size_pixels) - relative_y)
cv2.circle(improved_damier[0], (relative_point_x, relative_point_y), 5, (0, 255, 0), -1)

# Affichage de l'image avec les points
cv2.imshow("Image avec points", improved_damier[0])
cv2.waitKey(0)
cv2.destroyAllWindows()