import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from images_converter import Converter
from camera_calibrator import CameraCalibrator


class ImproveData:
    def __init__(self, checkerboard, diagonal_square_size, path, energy):
        """Classe pour améliorer les données d'images en enlevant l'arrière-plan et en redressant les images.

        Args:
            checkerboard (tuple): Un tuple contenant le nombre de coins du damier dans les directions x et y.
            diagonal_square_size (float): Taille de la diagonale d'un carré du damier en millimètres.
            path (str): Chemin du répertoire contenant les images.
            energy (str): Nom de l'énergie des images.
        """
        raw_images = glob.glob(f"{path}/{energy}/*")
        backgrounds = glob.glob(f"{path}/Background/*")
        calibration_images = glob.glob(f"{path}/Calibration/*")

        self.calibrator = CameraCalibrator(checkerboard, diagonal_square_size, calibration_images)
        self.raw_images = Converter(raw_images, checkerboard, diagonal_square_size).convert_fits2png()
        self.backgrounds = Converter(backgrounds, checkerboard, diagonal_square_size).convert_fits2png()

        self.path = path
        self.energy = energy
        self.undistorted_images = []
        self.cleaned_images = {}

    def get_file_names(self):
        """Récupère les noms de fichiers des images.

        Returns:
            list: Liste des noms de fichiers des images sans extension.
        """
        try:
            folder_path = f"{self.path}/{self.energy}"
            extension = ".fit"
            file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            file_names_without_extension = [f[:-len(extension)] if f.endswith(extension) else f for f in file_names]
            return file_names_without_extension
        except OSError as e:
            print(f"Error accessing folder: {e}")
            return []

    def remove_background(self):
        """Supprime l'arrière-plan des images brutes.

        Raises:
            ValueError: Si les images n'ont pas les mêmes dimensions.

        Returns:
            list: Liste des images sans l'arrière-plan.
        """
        matrice_bg = np.dstack((self.backgrounds))
        mean_bg = np.mean(matrice_bg, axis=2).astype(np.uint16)
        for i, noisy_image in enumerate(self.raw_images):
            # Vérifier que les images ont les mêmes dimensions
            if noisy_image.shape != mean_bg.shape:
                raise ValueError("The images do not have the same dimensions.")
            # Soustraction de l'image de background de l'image avec bruit
            cleaned_image = cv.absdiff(noisy_image, mean_bg)
            file_name = self.get_file_names()[i]
            self.cleaned_images[file_name] = cleaned_image
        return self.cleaned_images

    def radiative_noise(self):
        backgroundless = self.remove_background()
        prefix_dict = {}
        for filename in backgroundless.keys():
            parts = filename.split("_")
            if parts[0] not in prefix_dict:
                prefix_dict[parts[0]] = []
            prefix_dict[parts[0]].append(filename)

        # Calculer la médiane pour chaque clé dans le dictionnaire
        median_dict = {}
        for prefix, filenames in prefix_dict.items():
            arrays_to_median = [backgroundless[filename] for filename in filenames]
            median_array = np.median(arrays_to_median, axis=0).astype(arrays_to_median[0].dtype)
            median_dict[prefix] = median_array
        return median_dict

    def plot_colormap(self, colormap_name="viridis"):
        for name, image_data in self.radiative_noise().items():
            _, ax = plt.subplots()
            im = ax.imshow(image_data, cmap=colormap_name)
            cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
            ax.tick_params(left=False, bottom=False, labelleft = False ,
                labelbottom = False)
            cbar.set_label('Gray Value', fontsize=16)
            ax.text(0.03, 0.98, self.energy, transform=ax.transAxes,
                fontsize=12, color='black',
                bbox=dict(facecolor="w", edgecolor='k', boxstyle='round,pad=0.4'))
            plt.savefig(f"{self.path}/Improved_Data/{self.energy}/colormap_{name}.png",
                bbox_inches ="tight", dpi=600, transparent=True)

    def straighten_image(self):
        """Redresse les images en corrigeant la distorsion.

        Returns:
            list: Liste des images redressées et corrigées de la distorsion
        """
        _, mtx, dist, _, _ = self.calibrator.calibrate_camera()
        for dist_image in self.remove_background():
            framesize = dist_image.shape[:2]
            # Obtenir une nouvelle matrice de caméra optimale et une région d'intérêt pour l'image corrigée
            newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, framesize[::-1], 1,  framesize[::-1])
            # Appliquer la correction de distorsion à l'image
            undistort = cv.undistort(dist_image, mtx, dist, None, newCameraMatrix)
            # Recadrer l'image à la région d'intérêt (ROI)
            x, y, w, h = roi
            self.undistorted_images.append(undistort[y:y+h, x:x+w])
        return self.undistorted_images

    def see_raw_images(self):
        """Enregistre les images brute dans un répertoire 'Raw_Data'.
        """
        self.radiative_noise()
        directory = f"{self.path}/Raw_Data/{self.energy}"

        if not os.path.exists(directory):
            os.makedirs(directory)

        for count, raw_image in enumerate(self.raw_images):
            cv.imwrite("{0}/{1}.png".format(directory, self.get_file_names()[count]), raw_image)

    def improve_data(self, straight=False, median_filter=False, colormap=False):
        """Améliore les images en supprimant l'arrière-plan et les redressant.
        Les images améliorées sont enregistrées dans un répertoire 'Improved_Data'.
        """
        directory = f"{self.path}/Improved_Data/{self.energy}"

        if not os.path.exists(directory):
            os.makedirs(directory)

        images = self.straighten_image() if straight else self.radiative_noise()

        if isinstance(images, dict):
            for name, improved_image in images.items():
                if median_filter:
                    improved_image = cv.medianBlur(improved_image, ksize=3)
                elif colormap:
                    self.plot_colormap()
                cv.imwrite("{0}/{1}.png".format(directory, name), improved_image)

        else:
            for count, improved_image in enumerate(images):
                cv.imwrite("{0}/{1}.png".format(directory, self.get_file_names()[count]), improved_image)