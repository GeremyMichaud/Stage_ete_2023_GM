"""import cv2 as cv
import numpy as np

# Charger l'image en niveaux de gris 16-bit
image_beam = cv.imread("Measurements/2023-07-24/Improved_Data/6MV/3.png", cv.IMREAD_UNCHANGED)
"Measurements/2023-07-10/Improved_Data/18MV/unpol.png"

# Créer un masque pour les pixels à rendre transparents
mask = np.where(image_beam < 4000, 0, 65535).astype(np.uint16)

# Créer un canal alpha en utilisant le masque
alpha_channel = mask

# Fusionner l'image originale avec le canal alpha
image_rgba = cv.merge((image_beam, image_beam, image_beam, alpha_channel))

# Sauvegarder l'image modifiée au format 16-bit PNG
cv.imwrite("Measurements/2023-07-10/Improved_Data/18MV/image_transparente.png", image_rgba)"""

import cv2 as cv
import numpy as np

# Charger l'image en niveaux de gris 16-bit
image_beam = cv.imread("Measurements/2023-07-24/Improved_Data/6MV/3.png", cv.IMREAD_UNCHANGED)
"Measurements/2023-07-10/Improved_Data/18MV/unpol.png"

# Définir les coordonnées du rectangle centré (x, y, largeur, hauteur)
center_x = image_beam.shape[1] // 2  # moitié de la largeur
center_y = image_beam.shape[0] // 2  # moitié de la hauteur
rect_width = 200  # Largeur du rectangle centré
rect_height = 770  # Hauteur du rectangle centré

# Créer un masque pour les pixels en dehors du rectangle centré
threshold = 3000
mask = np.zeros_like(image_beam, dtype=np.uint16)
mask[center_y - rect_height // 2:center_y + rect_height // 2, center_x - rect_width // 2:center_x + rect_width // 2] = np.where(image_beam[center_y - rect_height // 2:center_y + rect_height // 2, center_x - rect_width // 2:center_x + rect_width // 2] < threshold, 0, 65535).astype(np.uint16)

# Créer un canal alpha en utilisant le masque
alpha_channel = mask

# Fusionner l'image originale avec le canal alpha
image_rgba = cv.merge((image_beam, image_beam, image_beam, alpha_channel))

# Sauvegarder l'image modifiée au format 16-bit PNG
cv.imwrite("Measurements/2023-07-24/Cube/image_transparente.png", image_rgba)
