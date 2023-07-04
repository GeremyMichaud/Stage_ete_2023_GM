from astropy.io import fits
import cv2 as cv
import os

class Converter:
    def __init__(self, images, path=None):
        """Initialise un objet Converter.

        Args:
            images (list): Une liste contenant les FITS images à convertir.
            path (str, optional): Le chemin vers le dossier à vérifier. Par défaut, il est défini à None.
        """
        self.images = images
        self.path = path
        self.jpeg_images_data = []

    def convert_fits2jpeg(self):
        """Convertit une liste d'images FITS en images JPEG normalisées.

        Returns:
            list: Une liste contenant les données des images JPEG normalisées pour chaque image FITS.
        """
        for fits_path in self.images:
            fits_data = fits.open(fits_path)
            fits_image_data = fits_data[0].data
            self.jpeg_images_data.append(cv.normalize(fits_image_data, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U))
        return self.jpeg_images_data

    def verify_file_path(self):
        """Vérifie si le chemin d'accès de fichier existe.

        Raises:
            FileNotFoundError: Si le chemin spécifié n'existe pas.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"The path {self.path} does not exist.")