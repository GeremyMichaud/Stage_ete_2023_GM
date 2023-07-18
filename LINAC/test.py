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

# Taille réelle du carré en millimètres
real_square_size = np.sqrt( 25**2 / 2)  # Remplacez par la taille réelle du carré en millimètres

calib = glob.glob(f"Measurements/2023-06-27/Calibration/*")

wedge_finder = CameraCalibrator((damier_width, damier_height), calib)

_, corners = wedge_finder.find_chessboard_corners(wedge_finder.images[0])

distances = []

for i in range(len(corners) - 1):
    x1, y1 = corners[i].ravel().astype(int)
    x2, y2 = corners[i+1].ravel().astype(int)
    if y2 < y1 + 20:
        distance_pixels = (abs(x2 - x1)**2 + abs(y2 - y1)**2)** 0.5
        distances.append(distance_pixels)

if len(distances) > 0:
    avg_distances = np.mean(distances)
    std_distance = np.std(distances)
    print(f"La distance moyenne entre les coins sur la même ligne est de : {avg_distances:.3f} pixels.")
    print(f"L'écart-type des distances entre les coins est de : {std_distance:.3f} pixels.")
else:
    print("Aucune paire de coins consécutifs sur la même ligne n'a été trouvée.")



# Sélectionner un carré spécifique du damier (ici, le carré en haut à gauche)
target_square_index = 6
target_square_corner = corners[target_square_index]

# Extraire les coordonnées du coin supérieur gauche du carré cible
x, y = target_square_corner.ravel().astype(int)

# Calculer les coordonnées de l'axe central vertical
central_axis = image_damier[0].size // 2

# Créer une copie colorisée de l'image
image_color = cv2.cvtColor(image_damier, cv2.COLOR_GRAY2BGR)

# Dessiner un point sur le coin supérieur gauche du carré cible
cv2.circle(image_color, (x, y), 5, (0, 0, 255), -1)

# Dessiner une ligne verticale sur l'axe central
cv2.line(image_color, (central_axis, 0), (central_axis, image_color.shape[0]), (0, 255, 0), 1)

# Afficher l'image avec le point et la ligne
cv2.imshow("Image avec point et ligne", image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
