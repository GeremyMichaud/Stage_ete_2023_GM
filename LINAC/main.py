from camera_calibrator import CameraCalibrator
from images_converter import Converter
from images_improver import ImproveData
from image_analysis import Analysis
from japanese import Japanese
import glob


# Définir la taille du damier (nombre de coins intérieurs par ligne et colonne)
CHECKERBOARD = (7, 10)
DIAGONALE = 25

if __name__ == "__main__":
    # Définir la date pour extraire les images du dossier correspondant
    #date = input("Enter the date of your data acquision (YYYY-MM-DD):")
    #energy = input("Enter the energy level of your data acquision:")
    date = "2023-07-10"
    energy = "6MV"
    path = f"Measurements/{date}"

    calib = glob.glob(f"{path}/Calibration/*")

    # Créer les instances des classes
    converter = Converter(calib, CHECKERBOARD, DIAGONALE, path)
    calibrator = CameraCalibrator(CHECKERBOARD, DIAGONALE, calib)
    improved = ImproveData(CHECKERBOARD, DIAGONALE, path, energy)
    japan = Japanese(CHECKERBOARD, DIAGONALE, path, energy)
    analyse = Analysis(CHECKERBOARD, DIAGONALE, path, energy)

    # Vérifier le chemin d'accès
    try:
        converter.verify_file_path()
    except FileNotFoundError as e:
        print(e)

    #improved.improve_data(colormap=True)
    #improved.see_raw_images()

    #.improve_data()
    #japan.polarizing_component()
    #japan.non_polarized()
    japan.plot_pdd()
    japan.verif_factor()

    # Appeler les méthodes de calibration de la caméra
    #calibrator.print_calib_coeff()
    #calibrator.show_chessboard_corners()
    #calibrator.reprojection_error()
    #calibrator.undistort_calibration_images()

    # Appeler les méthodes de conversion de pixel à mm
    #converter.print_pixel2mm_factors(0)
    #converter.calib_show_central_axis(5, 0)

    # Appeler les méthodes d'analyse d'image
    #analyse.plot_profile()
    #analyse.plot_pdd()