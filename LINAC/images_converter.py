from astropy.io import fits
import cv2 as cv
import numpy as np
import os

class Converter:
    def __init__(self, images, checkerboard, path=None):
        """Initialise un objet Converter.

        Args:
            images (list): Une liste contenant les FITS images à convertir.
            checkerboard (tuple): Un tuple contenant le nombre de coins du damier dans les directions x et y.
            path (str, optional): Le chemin vers le dossier à vérifier. Par défaut, il est défini à None.
        """
        self.images = images
        self.checkerboard = checkerboard
        self.path = path
        self.png_images_data = []

    def convert_fits2png(self):
        """Convertit une liste d'images FITS en images png normalisées.

        Returns:
            list: Une liste contenant les données des images png normalisées pour chaque image FITS.
        """
        for fits_path in self.images:
            fits_data = fits.open(fits_path)
            fits_image_data = fits_data[0].data
            self.png_images_data.append(cv.normalize(fits_image_data, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U))
        return self.png_images_data

    def verify_file_path(self):
        """Vérifie si le chemin d'accès de fichier existe.

        Raises:
            FileNotFoundError: Si le chemin spécifié n'existe pas.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"The path {self.path} does not exist.")

    def convert_pixel2mm(self, diagonal_square_size):
        """Convertit les pixels en millimètres en utilisant les coins d'un damier trouvé dans l'image.

        Args:
            diagonal_square_size (float or int): Taille de la diagonale d'un carré du damier en mm.

        Raises:
            ValueError: Si aucun damier n'est trouvé dans l'image ou si aucune paire de coins consécutifs
                sur la même ligne n'est trouvée.

        Returns:
            tuple: Un tuple contenant le facteur de conversion pixel vers mm, la distance moyenne entre les coins
                du damier en pixels et l'écart-type des distances entre les coins en pixels.
        """
        # Taille réelle du carré en millimètres
        real_square_size = np.sqrt(diagonal_square_size ** 2 / 2)
        image_data = self.convert_fits2png()[0]
        ret, corners = cv.findChessboardCorners(image_data, self.checkerboard, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        if not ret:
            raise ValueError("No checkerboard found in the image.")

        # Spécifier les paramètres pour les critères d'arrêt
        # Dans ce cas, 30 indique le nombre maximal d'itérations autorisées et 0.001 indique la précision (epsilon) à atteindre
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Affiner les coordonnées des coins trouvés pour une meilleure précision
        corners = cv.cornerSubPix(image_data, corners, (11, 11), (-1, -1), criteria)

        distances = []
        for i in range(len(corners) - 1):
            x1, y1 = corners[i].ravel().astype(int)
            x2, y2 = corners[i+1].ravel().astype(int)
            if y2 < y1 + 20:
                distance_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distances.append(distance_pixels)

        if len(distances) == 0:
            raise ValueError("No consecutive corner pairs on the same line were found.")

        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        pixel2mm_factor = real_square_size / avg_distance

        return pixel2mm_factor, avg_distance, std_distance

    def print_pixel2mm_factors(self, diagonal_square_size):
        """Affiche les facteurs de conversion de pixels en millimètres, la distance moyenne entre les coins du damier
        en pixels, l'écart-type des distances entre les coins en pixels et le nombre de pixels par millimètre.

        Args:
            diagonal_square_size (float or int): Taille de la diagonale d'un carré du damier en mm.
        """
        factors = self.convert_pixel2mm(diagonal_square_size)
        real_square_size = np.sqrt(diagonal_square_size ** 2 / 2)
        pixel_per_mm = 1 / factors[0]

        print(f"The average distance between consecutive corners on the same line is: {factors[1]:.3f} pixels.")
        print(f"The standard deviation of distances between checkerboard corners is: {factors[2]:.3f} pixels.")
        print(f"Considering the actual distance between 2 consecutive corners is {real_square_size:.3f} mm,",
            f"this means there are {pixel_per_mm:.3f} pixels per mm.")