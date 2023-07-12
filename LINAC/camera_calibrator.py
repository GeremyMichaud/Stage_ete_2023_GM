import cv2 as cv
import numpy as np
from images_converter import Converter


class CameraCalibrator:
    def __init__(self, checkerboard, fits_images):
        """Initialiser le calibrateur de caméra avec le damier spécifié.

        Args:
            checkerboard (tuple): Un tuple contenant le nombre de coins du damier dans les directions x et y.
            fits_images (list): Liste des données d'images FITS utilisées pour la calibration.
        """
        self.checkerboard = checkerboard
        self.objpoints = []
        self.imgpoints = []

        # Définir les coordonnées 3D réelles des coins du damier
        objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

        self.objp = objp

        converter = Converter(fits_images)
        self.images = converter.convert_fits2png()

    def find_chessboard_corners(self, image_data):
        """Trouver les coins du damier sur une seule image.

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

    def calibrate_camera(self):
        """Effectuer la calibration de la caméra.

        Returns:
            ret (bool) : True si la calibration est réussie, False sinon.
            mtx (numpy.ndarray) : Matrice de la caméra (matrice intrinsèque).
            dist (numpy.ndarray) : Coefficients de distorsion.
            rvecs (list) : Liste des vecteurs de rotation pour chaque image.
            tvecs (list) : Liste des vecteurs de translation pour chaque image.
        """
        for png_image_data in self.images:
            # Trouver les coins du damier sur l'image actuelle
            ret, corners = self.find_chessboard_corners(png_image_data)
            if ret:
                # Si des coins sont trouvés, ajouter les coordonnées 3D et 2D aux vecteurs objpoints et imgpoints
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)

        framesize = png_image_data.shape[:2]  # 1392 X 1040 pixels

        try:
            # Effectuer la calibration de la caméra en utilisant les vecteurs objpoints et imgpoints
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, framesize[::-1], None, None)
        except cv.error as error:
            print(f"Error while calibrating the camera:\n{error}")
            return False, None, None, None, None
        return ret, mtx, dist, rvecs, tvecs

    def print_calib_coeff(self):
        """Afficher les coefficients de calibration de la caméra.
        """
        ret, mtx, dist, rvecs, tvecs = self.calibrate_camera()

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

    def reprojection_error(self):
        """Calculer et afficher l'erreur de reprojection moyenne pour la calibration de la caméra.
        """
        mean_error = 0
        _, mtx, dist, rvecs, tvecs = self.calibrate_camera()
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print("Reprojection error : {}".format(mean_error/len(self.objpoints)))

    def show_chessboard_corners(self):
        """Afficher les coins du damier sur les images données.
        """
        for png_image_data in self.images:
            # Trouver les coins du damier sur l'image actuelle
            ret, corners = self.find_chessboard_corners(png_image_data)
            if ret:
                # Dessiner et afficher les coins trouvés sur l'image
                img = cv.drawChessboardCorners(png_image_data, self.checkerboard, corners, ret)
                cv.imshow("Detected Corners", img)
                cv.waitKey(2000)  # Afficher l'image pendant 2000 millisecondes
        cv.destroyAllWindows()

    def undistort_calibration_images(self):
        """Corriger la distortion des images données en utilisant la calibration de la caméra
        et afficher les images de calibration corrigées.
        """
        _, mtx, dist, _, _ = self.calibrate_camera()

        for png_image_data in self.images:
            framesize = png_image_data.shape[:2]
            # Obtenir une nouvelle matrice de caméra optimale et une région d'intérêt (ROI) pour l'image corrigée
            newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, framesize[::-1], 1,  framesize[::-1])
            # Appliquer la correction de distorsion à l'image
            undistort = cv.undistort(png_image_data, mtx, dist, None, newCameraMatrix)
            # Recadrer l'image à la région d'intérêt (ROI)
            x, y, w, h = roi
            cv.imshow("Undistorted calibration images", undistort[y:y+h, x:x+w])
            cv.waitKey(2000)  # Afficher l'image pendant 2000 millisecondes
        cv.destroyAllWindows()
