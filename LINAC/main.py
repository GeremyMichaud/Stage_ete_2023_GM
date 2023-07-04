from camera_calibrator import CameraCalibrator
from images_converter import Converter
from images_improver import ImproveData
import glob
import os
import cv2 as cv


# Définir la taille du damier (nombre de coins intérieurs par ligne et colonne)
CHECKERBOARD = (7, 10)

if __name__ == "__main__":
    # Définir la date pour extraire les images du dossier correspondant
    #date = input("Enter the date of your data acquision (YYYY-MM-DD):")
    date = "2023-06-27"
    path = f"Measurements/{date}"

    calib = glob.glob(f"{path}/Calibration/*")
    back = glob.glob(f"{path}/Background/*")
    energy = "6MV"

    # Créer les instances des classes
    converter = Converter(calib, path)
    calibrator = CameraCalibrator(CHECKERBOARD, calib)
    improved = ImproveData(CHECKERBOARD, path, energy, back, calib)

    # Vérifier le chemin d'accès
    try:
        converter.verify_file_path()
    except FileNotFoundError as e:
        print(e)
    improved.improve_data()

    # Appeler les méthodes de calibration de la caméra
    #calibrator.print_calib_coeff()
    #calibrator.show_chessboard_corners()
    #calibrator.reprojection_error()
    #calibrator.undistort_calibration_images()