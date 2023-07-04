import cv2 as cv
import os
from images_converter import Converter
from camera_calibrator import CameraCalibrator


class ImproveData:
    def __init__(self, checkerboard, path, raw_images, background, calibration_images):
        self.raw_images = Converter(raw_images).convert_fits2jpeg()
        self.background = Converter(background).convert_fits2jpeg()
        self.path = path
        self.undistorted_images = []
        self.cleaned_images = []
        self.calibrator = CameraCalibrator(checkerboard, calibration_images)

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

    def improve_data(self):
        directory = f"{self.path}/Improved_Data"

        if not os.path.exists(directory):
            os.makedirs(directory)

        for count, improved_image in enumerate(self.straighten_image()):
            cv.imwrite("{0}/Calibrated_{1}.png".format(directory, count+1), improved_image)