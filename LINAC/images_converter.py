from astropy.io import fits
import cv2 as cv
import numpy as np
import os

class Converter:
    def __init__(self, images, checkerboard, diagonal_square_size, path=None):
        """Initialise un objet Converter.

        Args:
            images (list): Une liste contenant les chemins d'accès des images à convertir.
            checkerboard (tuple): Un tuple contenant le nombre de coins du damier dans les directions x et y.
            diagonal_square_size (float or int): Taille de la diagonale d'un carré du damier en milimètres.
            path (str, optional): Le chemin vers le dossier à vérifier. Par défaut, il est défini à None.
        """
        self.images = images
        self.checkerboard = checkerboard
        self.diagonal_square_size = diagonal_square_size
        self.path = path
        self.png_16bit_images_data = []

    def convert_fits2png(self):
        """Convertit une liste d'images FITS en images png normalisées.

        Returns:
            list: Une liste contenant les données des images png normalisées pour chaque image FITS.
        """
        for fits_path in self.images:
            fits_data = fits.open(fits_path)
            fits_image_data = fits_data[0].data
            # Convertir les images FITS en 16-bit PNG (range 0-65535)
            self.png_16bit_images_data.append(((fits_image_data - fits_image_data.min()) / (fits_image_data.max() - fits_image_data.min()) * 65535).astype(np.uint16))
        return self.png_16bit_images_data

    def verify_file_path(self):
        """Vérifie si le chemin d'accès de fichier existe.

        Raises:
            FileNotFoundError: Si le chemin spécifié n'existe pas.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"The path {self.path} does not exist.")

    def convert_pixel2mm(self, calib_image_index):
        """Convertit les pixels en millimètres en utilisant les coins d'un damier trouvé dans l'image.

        Args:
            calib_image_index (int): L'indice de l'image de calibration à montrer dans l'ordre qu'elle est dans le fichier d'origine.

        Raises:
            ValueError: Si aucun damier n'est trouvé dans l'image ou si aucune paire de coins consécutifs
                sur la même ligne n'est trouvée.

        Returns:
            tuple: Un tuple contenant le facteur de conversion pixel vers mm, la distance moyenne entre les coins
                du damier en pixels, l'écart-type des distances entre les coins en pixels et la position des coins en pixel.
        """
        # Taille réelle du carré en millimètres
        real_square_size = np.sqrt(self.diagonal_square_size ** 2 / 2)
        image_data = self.convert_fits2png()[calib_image_index]
        # Retransformer en 8-bit PNG, puisque le module findChessboardCorners ne fonctionne qu'avec ce type de données
        image_8bit = ((image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255).astype(np.uint8)
        ret, corners = cv.findChessboardCorners(image_8bit, self.checkerboard, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        if not ret:
            raise ValueError("No checkerboard found in the image.")

        # Spécifier les paramètres pour les critères d'arrêt
        # Dans ce cas, 30 indique le nombre maximal d'itérations autorisées et 0.001 indique la précision (epsilon) à atteindre
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Affiner les coordonnées des coins trouvés pour une meilleure précision
        corners = cv.cornerSubPix(image_8bit, corners, (11, 11), (-1, -1), criteria)

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

        return pixel2mm_factor, avg_distance, std_distance, corners

    def print_pixel2mm_factors(self, calib_image_index):
        """Affiche les facteurs de conversion de pixels en millimètres, la distance moyenne entre les coins du damier
        en pixels, l'écart-type des distances entre les coins en pixels et le nombre de pixels par millimètre.

        Args:
            calib_image_index (int): L'indice de l'image de calibration à montrer dans l'ordre qu'elle est dans le fichier d'origine.
        """
        factors = self.convert_pixel2mm(calib_image_index)
        real_square_size = np.sqrt(self.diagonal_square_size ** 2 / 2)
        pixel_per_mm = 1 / factors[0]

        print(f"The average distance between consecutive corners on the same line is: {factors[1]:.3f} pixels.")
        print(f"The standard deviation of distances between checkerboard corners is: {factors[2]:.3f} pixels.")
        print(f"Considering the actual distance between 2 consecutive corners is {real_square_size:.3f} mm,",
            f"this means there are {pixel_per_mm:.3f} pixels per mm.")

    def central_axis(self):
        """Calcule la coordonnée de l'axe central vertical de l'image.

        Returns:
            list: Une liste de la coordonnée de l'axe central vertical de chaque image du fichier.
        """
        images_data = self.convert_fits2png()
        central_axis = []

        for image in images_data:
            # Calculer les coordonnées de l'axe central vertical
            central_axis.append(image[0].size // 2)

        return central_axis

    def calib_show_central_axis(self, target_square_index, calib_image_index):
        """Affiche l'axe central vertical et un point sur le coin cible de l'image du damier.

        Args:
            target_square_index (int): L'indice du coin cible dans la liste des coins du damier.
            calib_image_index (int): L'indice de l'image de calibration à montrer dans l'ordre qu'elle est dans le fichier d'origine.
        """
        corners = self.convert_pixel2mm(calib_image_index)[3]
        target_square_corner = corners[target_square_index]
        central_axis = self.central_axis()[calib_image_index]

        # Extraire les coordonnées du coin cible
        x, y = target_square_corner.ravel().astype(int)

        # Normaliser l'image de 16 bits à une plage de 0-255 (8 bits)
        normalized_image = cv.normalize(self.convert_fits2png()[calib_image_index], None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

        # Convertir en couleur en utilisant un nuance de gris pour chaque canal
        image_color = cv.cvtColor(normalized_image, cv.COLOR_GRAY2BGR)

        # Dessiner un point sur le coin cible
        cv.circle(image_color, (x, y), 4, (0, 0, 255), -1)

        # Dessiner une ligne verticale sur l'axe central
        cv.line(image_color, (central_axis, 0), (central_axis, image_color.shape[0]), (0, 255, 0), 1)

        # Afficher l'image avec le point et la ligne
        cv.imshow("Target corner and center axis", image_color)
        cv.waitKey(0)
        cv.destroyAllWindows()
