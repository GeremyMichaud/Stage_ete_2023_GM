import cv2
from astropy.io import fits

# Load the .fit image
fit_image = fits.open("Measurements/2023-06-27/Calibration/1sec_0001.fit")

# Extract the image data from the FITS file
image_data = fit_image[0].data

# Normalize the pixel values to the 0-255 range
image_damier = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("Damier", image_damier)

# Application d'une opération de seuillage pour convertir en image binaire
_, threshold = cv2.threshold(image_damier, 128, 255, cv2.THRESH_BINARY)

# Recherche des contours dans l'image seuillée
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Trier les contours par aire, en supposant que le carré du damier aura une des plus grandes aires
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Sélection du contour avec la plus grande aire
plus_grand_contour = contours[0]

# Calculer le rectangle englobant du contour
x, y, w, h = cv2.boundingRect(plus_grand_contour)

# Dessiner le rectangle englobant sur l'image d'origine
image_rectangle = cv2.rectangle(image_damier.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)

# Afficher l'image avec le rectangle englobant
cv2.imshow("Image avec rectangle englobant", image_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()
