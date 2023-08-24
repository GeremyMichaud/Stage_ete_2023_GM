from images_converter import Converter
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob


class Analysis:
    def __init__(self, checkerboard, diagonal_square_size, path, energy):
        """Initialize the Analysis class.

        Args:
            checkerboard (tuple): A tuple containing the number of corners of the checkerboard in the x and y directions.
            diagonal_square_size (float or int): Size of the diagonal of a checkerboard square in mm.
            path (str): The path of the directory containing the images.
            energy (str): Image energy name.
        """
        # Initialize image paths for improved images and calibration images
        self.improved_images = glob.glob(f"{path}/Improved_Data/{energy}/*")
        calibration_images = glob.glob(f"{path}/Calibration/*")

        self.path = path
        self.energy = energy
        self.folder_path = os.path.join(path, "Improved_Data", energy)

        # Initialize pixel-to-mm converter using the provided calibration images
        self.pixel_converter = Converter(calibration_images, checkerboard, diagonal_square_size).convert_pixel2mm(
            calib_image_index=0)

    def get_file_names(self):
        """Retrieves image filenames.

        Returns:
            list: List of image filenames without extension.
        """
        try:
            extension = ".fit"
            # List files in the folder and extract file names without extensions
            file_names = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
            file_names_without_extension = [f[:-len(extension)] if f.endswith(extension) else f for f in file_names]
            return file_names_without_extension
        except OSError as e:
            print(f"Error accessing folder: {e}")
            return []

    def pdd_grayvalues(self):
        """Calculate intensity profiles along the vertical axis (percentage depth dose).

        Returns:
            tuple: Lists of intensity profiles, maximum values, and maximum indices.
        """
        interval_size = 20
        x_offset = 30
        intensity_list = []
        max_value_list =[]
        max_index_list = []

        for improved_image in self.improved_images:
            img = plt.imread(improved_image)
            width = img.shape[1]
            central_row = width // 2

            start_line = central_row - interval_size - x_offset
            end_line = central_row + interval_size - x_offset

            # Calculate the mean intensity along the vertical axis
            img_centered = []
            for row in img:
                img_centered.append(row[start_line:end_line])

            intensity = np.mean(img_centered, axis=1)
            intensity_list.append(list(intensity))
            max_index = np.argmax(intensity)
            max_value_list.append(intensity[max_index])
            max_index_list.append(max_index)

        return intensity_list, max_value_list, max_index_list, start_line, end_line

    def curve_fft(self, x_data_graph):
        """Perform Fourier transform and return the reconstructed data after thresholding.

        Args:
            x_data_graph (array-like): Input data for Fourier transform.

        Returns:
            array-like: Reconstructed data after thresholding.
        """
        # Calculate discrete Fourier transform (DFT)
        coeff_dft = np.fft.rfft(x_data_graph)
        # Calculate threshold to keep the first 10% of coefficients
        dft_thresholded_10 = coeff_dft.copy()
        ten_percent = int(0.1*len(dft_thresholded_10))
        # Apply threshold to zero out the remaining 90% coefficients
        dft_thresholded_10[ten_percent:] = 0
        # Perform inverse Fourier transform to reconstruct the data
        reconstructed_data = np.fft.irfft(dft_thresholded_10)
        return reconstructed_data

    def plot_pdd(self):
        """Plot Percentage Depth Dose (PDD) profiles.
        """
        intensity_profiles = self.pdd_grayvalues()[0]
        directory = os.path.join(self.path, "PDD", self.energy)
        os.makedirs(directory, exist_ok=True)
        target_position_cm = 1.5

        film_path = f"Measurements/Data_Emily/RP{self.energy}_film.txt"
        film_data = []
        for data in np.loadtxt(film_path):
            film_data.append(data)
        max_film_index = np.argmax(film_data)
        film_data_per_cm = max_film_index / target_position_cm
        film_total_cm = len(film_data) / film_data_per_cm
        film_position_cm = np.linspace(0, film_total_cm, len(film_data))

        for index, intensity in enumerate(intensity_profiles):
            relative_intensity = intensity / self.pdd_grayvalues()[1][index]
            reconstructed_data = self.curve_fft(relative_intensity)

            max_relative_intensity_index = np.argmax(relative_intensity)
            position_pix = np.linspace(0, len(reconstructed_data), len(reconstructed_data))
            position_cm = position_pix * self.pixel_converter[0] /10
            max_intensity_position_cm = position_cm[max_relative_intensity_index]
            offset = target_position_cm - max_intensity_position_cm

            # Create and customize the plot
            fig, ax = plt.subplots()
            palette = sns.color_palette("colorblind")
            ax.plot(position_cm + offset, reconstructed_data, color=palette[2], linewidth="1.5", label="Raw Cherenkov")
            ax.plot(film_position_cm, film_data, color=palette[4], linewidth="1.5", label="Radiochromic Film")
            plt.axvline(x=target_position_cm, color="black", linestyle=":", linewidth="1.2")
            ax.legend(loc="lower center", fontsize=16)
            ax.minorticks_on()
            ax.tick_params(top=True, right=True, axis="both", which="both", direction='in')
            ax.set_ylabel("Percentage depth dose [-]", fontsize=16)
            ax.set_xlabel("Depth [cm]", fontsize=16)
            ax.set_xlim(0, 16)

            numbers = "".join(filter(str.isdigit, self.energy))
            text = "".join(filter(str.isalpha, self.energy))
            ax.text(x=13, y=0.9, s="{0} {1}".format(numbers, text), fontsize=14)

            # Save the plot as an image
            plt.savefig(os.path.join(directory, "with_ref"), bbox_inches="tight", dpi=600)
            plt.close(fig)

    def plot_interval(self):
        for image in self.improved_images:
            img = cv.imread(image)
            start, end = self.pdd_grayvalues()[3], self.pdd_grayvalues()[4]
            interval_size = 20
            y_offset = 250
            height = img.shape[0]
            central_row = height // 2
            start_line = central_row - interval_size - y_offset
            end_line = central_row + interval_size - y_offset
            cv.line(img, (start, 0), (start, img.shape[0]), (0,255, 0), thickness=2)
            cv.line(img, (end, 0), (end, img.shape[0]), (0,255, 0), thickness=2)
            cv.line(img, (0, start_line), (img.shape[1], start_line), (0,255, 0), thickness=2)
            cv.line(img, (0, end_line), (img.shape[1], end_line), (0,255, 0), thickness=2)
            cv.imwrite('IGL.png', img)
            cv.destroyAllWindows()

    def calculate_curve_difference(self):
        # Get intensity profiles
        intensity_profiles = self.pdd_grayvalues()[0]
        target_position_cm = 1.5
        film_path = f"Measurements/Data_Emily/RP{self.energy}_film.txt"
        film_data = []
        for data in np.loadtxt(film_path):
            film_data.append(data)
        max_film_index = np.argmax(film_data)
        film_data_per_cm = max_film_index / target_position_cm
        film_total_cm = len(film_data) / film_data_per_cm
        film_position_cm = np.linspace(0, film_total_cm, len(film_data))

        for index, intensity in enumerate(intensity_profiles):
            relative_intensity = intensity / self.pdd_grayvalues()[1][index]
            reconstructed_data = self.curve_fft(relative_intensity)

            max_relative_intensity_index = np.argmax(relative_intensity)
            position_pix = np.linspace(0, len(reconstructed_data), len(reconstructed_data))
            position_cm = position_pix * self.pixel_converter[0] /10
            max_intensity_position_cm = position_cm[max_relative_intensity_index]
            offset = target_position_cm - max_intensity_position_cm
            position_centered = position_cm + offset

            # Restrict the data to the interval from 0 cm to 16 cm
            mask = (position_centered >= 0) & (position_centered <= 16)
            position_centered = position_centered[mask]
            reconstructed_data = reconstructed_data[mask]

            film_data = np.array(film_data)
            film_mask = (film_position_cm >= 0) & (film_position_cm <= 16)
            film_position_cm = film_position_cm[film_mask]
            film_data = film_data[film_mask]

            # Interpolate the reconstructed curve to match the film data
            interpolated_reconstructed_curve = np.interp(film_position_cm, position_centered, reconstructed_data)
            plt.plot(interpolated_reconstructed_curve)

            # Calculate the difference between the film data and the interpolated curve
            curve_difference = np.abs(film_data - interpolated_reconstructed_curve)

            # Calculate the average difference for this curve
            average_difference = np.mean(curve_difference)
            standard_deviation = np.std(curve_difference)

        return average_difference, standard_deviation

CHECKERBOARD = (7, 10)
DIAGONALE = 25
date = "2023-06-27"
energy = "6MV"
path = f"Measurements/{date}"
analyse = Analysis(CHECKERBOARD, DIAGONALE, path, energy)
#difference = analyse.calculate_curve_difference()
#print(f"Curve Average Difference: {difference[0]:.3e} ± {difference[1]:.3e}")
analyse.plot_pdd()
#analyse.plot_interval()