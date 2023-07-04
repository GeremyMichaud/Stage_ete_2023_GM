import cv2 as cv
import numpy as np
import glob
from astropy.io import fits
import os


class CameraCalibrator:
    def __init__(self, checkerboard):
        """Initialiser le calibrateur de caméra avec le damier spécifié.

        Args:
            checkerboard (tuple): Un tuple contenant le nombre de coins du damier dans les directions x et y.
        """
        self.checkerboard = checkerboard
        self.objpoints = []
        self.imgpoints = []

        # Définir les coordonnées 3D réelles des coins du damier
        objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

        self.objp = objp

    def find_chessboard_corners(self, image_data):
        """Trouver les coins du damier sur une seule image.

        Args:
            image_data (numpy.ndarray): Données de l'image à traiter

        Returns:
            ret (bool) : True si les coins du damier sont trouvés, False sinon.
            corners (numpy.ndarray) : Tableau contenant les coordonnées des coins du damier.
        """
        ret, corners = cv.findChessboardCorners(image_data, self.checkerboard, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            # Spécifier les paramètres pour les critères d'arrêt
            # Dans ce cas, 30 indique le nombre maximal d'itérations autorisées et 0.001 indique la précision (epsilon) à atteindre
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # Affiner les coordonnées des coins trouvés pour une meilleure précision
            corners = cv.cornerSubPix(image_data, corners, (11, 11), (-1, -1), criteria)
        return ret, corners

    def convert_fits2jpeg(self, images):
        """Convertit une liste d'images FITS en images JPEG normalisées.

        Args:
            images (list): Une liste contenant les données des images FITS.

        Returns:
            list: Une liste contenant les données des images JPEG normalisées pour chaque image FITS.
        """
        jpeg_image_data = []
        for fits_path in images:
            fits_data = fits.open(fits_path)
            fits_image_data = fits_data[0].data
            jpeg_image_data.append(cv.normalize(fits_image_data, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U))
        return jpeg_image_data

    def calibrate_camera(self, images):
        """Effectuer la calibration de la caméra.

        Args:
            images (list): Liste des données d'images JPEG normaliséess utilisées pour la calibration.

        Returns:
            ret (bool) : True si la calibration est réussie, False sinon.
            mtx (numpy.ndarray) : Matrice de la caméra (matrice intrinsèque).
            dist (numpy.ndarray) : Coefficients de distorsion.
            rvecs (list) : Liste des vecteurs de rotation pour chaque image.
            tvecs (list) : Liste des vecteurs de translation pour chaque image.
        """
        jpeg_images = self.convert_fits2jpeg(images)
        for jpeg_image_data in jpeg_images:
            # Trouver les coins du damier sur l'image actuelle
            ret, corners = self.find_chessboard_corners(jpeg_image_data)
            if ret:
                # Si des coins sont trouvés, ajouter les coordonnées 3D et 2D aux vecteurs objpoints et imgpoints
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)

        framesize = jpeg_image_data.shape[:2]  # 1392 X 1040 pixels

        try:
            # Effectuer la calibration de la caméra en utilisant les vecteurs objpoints et imgpoints
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, framesize[::-1], None, None)
        except cv.error as error:
            print(f"Error while calibrating the camera:\n{error}")
            return False, None, None, None, None
        return ret, mtx, dist, rvecs, tvecs

    def print_calib_coeff(self, images):
        """Afficher les coefficients de calibration de la caméra.

        Args:
            images (list): Liste des données d'images JPEG normalisées utilisées pour la calibration.
        """
        ret, mtx, dist, rvecs, tvecs = self.calibrate_camera(images)

        if ret:
            # Afficher les résultats de la calibration
            print(f"Overall RMS reprojection error : {ret}")
            print(f"\nCamera Matrix : \n{mtx}")
            print(f"\nFocal lenght in x direction : {mtx[0,0]}")
            print(f"Focal lenght in y direction : {mtx[1,1]}")
            print(f"Optical center : ({mtx[0,2]}, {mtx[1,2]})")
            print(f"\nDistortion Parameters : \n{dist}")
            print(f"\nRotation Vectors :")
            for rvec in rvecs:
                print(rvec.tolist())
            print(f"\nTranslation Vectors :")
            for tvec in tvecs:
                print(tvec.tolist())
        else:
            print("Calibration failed.")

    def reprojection_error(self, images):
        """Calcule l'erreur de reprojection moyenne pour la calibration de la caméra.

        Args:
            images (list): Liste des données d'images JPEG normalisées utilisées pour la calibration.
        """
        mean_error = 0
        _, mtx, dist, rvecs, tvecs = self.calibrate_camera(images)
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print("Reprojection error : {}".format(mean_error/len(self.objpoints)))

    def show_chessboard_corners(self, images):
        """Afficher les coins du damier sur les images données.

        Args:
            images (list): Liste des données d'images JPEG normalisées.
        """
        jpeg_images = self.convert_fits2jpeg(images)
        for jpeg_image_data in jpeg_images:
            # Trouver les coins du damier sur l'image actuelle
            ret, corners = self.find_chessboard_corners(jpeg_image_data)
            if ret:
                # Dessiner et afficher les coins trouvés sur l'image
                img = cv.drawChessboardCorners(jpeg_image_data, self.checkerboard, corners, ret)
                cv.imshow("Detected Corners", img)
                cv.waitKey(2000)  # Afficher l'image pendant 2000 millisecondes
        cv.destroyAllWindows()

    def undistort_images(self, images):
        """Corriger la distortion des images données en utilisant la calibration de la caméra.

        Args:
            images (list): Liste des données d'images JPEG normalisées.

        Returns:
            list: Une liste contenant les données des images corrigées de la distorsion.
        """
        jpeg_images = self.convert_fits2jpeg(images)
        _, mtx, dist, _, _ = self.calibrate_camera(images)
        undistorted_images = []

        for jpeg_image_data in jpeg_images:
            framesize = jpeg_image_data.shape[:2]
            # Obtenir une nouvelle matrice de caméra optimale et une région d'intérêt (ROI) pour l'image corrigée
            newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, framesize[::-1], 1,  framesize[::-1])
            # Appliquer la correction de distorsion à l'image
            undistort = cv.undistort(jpeg_image_data, mtx, dist, None, newCameraMatrix)
            # Recadrer l'image à la région d'intérêt (ROI)
            x, y, w, h = roi
            undistorted_images.append(undistort[y:y+h, x:x+w])
        return undistorted_images

if __name__ == "__main__":
    # Définir la taille du damier (nombre de coins intérieurs par ligne et colonne)
    CHECKERBOARD = (7, 10)
    # Définir la date pour extraire les images du dossier correspondant
    # date = input("Enter the date of your data acquision (YYYY-MM-DD):")
    date = "2023-06-27"
    images = glob.glob(f"Measurements/{date}/Calibration/*")

    # Créer une instance de la classe CameraCalibrator
    calibrator = CameraCalibrator(CHECKERBOARD)

    # Appeler les méthodes de calibration de la caméra
    #calibrator.print_calib_coeff(images)
    #calibrator.show_chessboard_corners(images)
    #calibrator.reprojection_error(images)
    undist = calibrator.undistort_images(images)
    directory = "Measurements/{0}/Undistorted".format(date)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for count, image in enumerate(undist):
        cv.imwrite("{0}/Calibrated_Chessboard_{1}.jpg".format(directory, count+1), image)
