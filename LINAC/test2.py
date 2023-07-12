import cv2
from astropy.io import fits

# Charger l'image FITS en niveaux de gris
fit_image = fits.open("Measurements/2023-06-27/Calibration/1sec_0001.fit")
image_data = fit_image[0].data
image_damier = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Dimensions du damier (nombre de carrés en largeur et en hauteur)
damier_width = 7
damier_height = 10

# Taille réelle du carré en millimètres
square_size_real = 20  # Remplacez par la taille réelle du carré en millimètres

# Trouver les coins du damier dans l'image
_, corners = cv2.findChessboardCorners(image_damier, (damier_width, damier_height))

# Sélectionner un carré spécifique du damier (ici, le carré en haut à gauche)
target_square_index = 2
target_square_corner = corners[target_square_index]

# Sélectionner deux coins adjacents
corner1_index = 0  # Indice du premier coin
corner2_index = 1  # Indice du deuxième coin (adjacent au premier coin)

# Extraire les coordonnées des deux coins
x1, y1 = corners[corner1_index].ravel().astype(int)
x2, y2 = corners[corner2_index].ravel().astype(int)

# Calculer la taille du carré en pixels
square_size_pixels = int(square_size_real * image_damier.shape[1] / (damier_width - 1))

# Calculer la distance en pixels entre les deux coins
distance_pixels = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# Afficher la distance en pixels entre les deux coins
print("Distance en pixels entre les deux coins adjacents :", distance_pixels)

# Extraire les coordonnées du coin supérieur gauche du carré cible
x, y = target_square_corner.ravel().astype(int)

# Calculer la taille du carré en pixels
square_size_pixels = int(square_size_real * image_damier.shape[1] / (damier_width - 1))

# Calculer les coordonnées de l'axe central vertical
central_axis_x = (damier_width - 1) // 2
central_axis_y = (damier_height - 1) // 2
central_axis_x_pos = central_axis_x * square_size_pixels

# Créer une copie colorisée de l'image
image_color = cv2.cvtColor(image_damier, cv2.COLOR_GRAY2BGR)

# Dessiner un point sur le coin supérieur gauche du carré cible
cv2.circle(image_color, (x, y), 5, (0, 0, 255), -1)

# Dessiner une ligne verticale sur l'axe central
cv2.line(image_color, (central_axis_x_pos, 0), (central_axis_x_pos, image_color.shape[0]), (0, 255, 0), 1)

# Afficher l'image avec le point et la ligne
cv2.imshow("Image avec point et ligne", image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
