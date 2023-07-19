import cv2 as cv
import os
import glob
from images_converter import Converter
from camera_calibrator import CameraCalibrator


class ImproveData:
    def __init__(self, checkerboard, path, energy):
        raw_images = glob.glob(f"{path}/{energy}/*")
        background = glob.glob(f"{path}/Background/*")
        calibration_images = glob.glob(f"{path}/Calibration/*")

        self.calibrator = CameraCalibrator(checkerboard, calibration_images)
        self.raw_images = Converter(raw_images, checkerboard).convert_fits2png()
        self.background = Converter(background, checkerboard).convert_fits2png()

        self.path = path
        self.energy = energy
        self.undistorted_images = []
        self.cleaned_images = []

    def get_file_names(self):
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
        for noisy_image in self.raw_images:
            # Vérifier que les images ont les mêmes dimensions
            if noisy_image.shape != self.background[0].shape:
                raise ValueError("The images do not have the same dimensions.")

            # Soustraction de l'image de background de l'image avec bruit
            self.cleaned_images.append(cv.absdiff(noisy_image, self.background[0]))
        return self.cleaned_images

    def straighten_image(self):
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
        directory = f"{self.path}/Raw_Data/{self.energy}"

        if not os.path.exists(directory):
            os.makedirs(directory)

        for count, raw_image in enumerate(self.raw_images):
            cv.imwrite("{0}/{1}.png".format(directory, self.get_file_names()[count]), raw_image)

    def improve_data(self):
        directory = f"{self.path}/Improved_Data/{self.energy}"

        if not os.path.exists(directory):
            os.makedirs(directory)

        for count, improved_image in enumerate(self.straighten_image()):
            cv.imwrite("{0}/{1}.png".format(directory, self.get_file_names()[count]), improved_image)