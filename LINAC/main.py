from cameracalibrator import CameraCalibrator
import cv2 as cv
import glob
import os


# Définir la taille du damier (nombre de coins intérieurs par ligne et colonne)
CHECKERBOARD = (7, 10)

def verify_file_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")

if __name__ == "__main__":
    # Définir la date pour extraire les images du dossier correspondant
    #date = input("Enter the date of your data acquision (YYYY-MM-DD):")
    date = "2023-06-27"
    path = f"Measurements/{date}"
    try:
        verify_file_path(path)
    except FileNotFoundError as e:
        print(e)

    images = glob.glob(f"{path}/Calibration/*")

    # Créer une instance de la classe CameraCalibrator
    calibrator = CameraCalibrator(CHECKERBOARD)

    # Appeler les méthodes de calibration de la caméra
    calibrator.print_calib_coeff(images)
    calibrator.show_chessboard_corners(images)
    calibrator.reprojection_error(images)
    undist = calibrator.undistort_images(images)
    directory = f"{path}/Undistorted"

    if not os.path.exists(directory):
        os.makedirs(directory)

    for count, image in enumerate(undist):
        cv.imwrite("{0}/Calibrated_Chessboard_{1}.png".format(directory, count+1), image)