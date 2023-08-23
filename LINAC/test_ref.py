from images_converter import Converter
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from scipy.signal import correlate


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
        self.improved_images = (f"{path}/Improved_Data/{energy}/unpol.png")
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

    def profile_grayvalues(self):
        """Calculate intensity profiles along a central horizontal interval.

        Returns:
            tuple: Lists of intensity profiles, maximum values, maximum indices, start and end lines.
        """
        interval_size = 45
        y_offset = 200
        intensity_list = []
        max_value_list =[]
        max_index_list = []

        img = plt.imread(self.improved_images)
        height = img.shape[0]
        central_row = height // 2

        start_line = central_row - interval_size - y_offset
        end_line = central_row + interval_size - y_offset

        # Calculate the mean intensity along the interval
        intensity = np.mean(img[start_line:end_line], axis=0)
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

    def central_axis(self, relative_intensity):
        """Find the central axis position based on relative intensity.

        Args:
            relative_intensity (array-like): Relative intensity data.

        Returns:
            int: Index of the central axis position.
        """
        range_plateau = max(relative_intensity) - 0.2
        max_index = np.argmax(relative_intensity)
        left_half = relative_intensity[:max_index]
        right_half = relative_intensity[max_index + 1:]

        # Find left and right points where intensity drops below plateau range
        left_point = next((len(left_half) - i - 1 for i, intensity in enumerate(left_half[::-1]) if intensity <= range_plateau), max_index)
        right_point = next((max_index + i + 1 for i, intensity in enumerate(right_half) if intensity <= range_plateau), max_index)

        # Calculate median index between the left and right points
        median_index = (left_point + right_point) // 2
        return median_index

    def plot_profile(self):
        """Plot intensity profiles along the off-axis direction.

        Args:
            plot_interval (bool, optional):  Whether to plot interval markers.. Defaults to False.
        """
        intensity_profiles = self.profile_grayvalues()[0]
        directory = os.path.join(self.path, "Profile", self.energy)
        os.makedirs(directory, exist_ok=True)

        film_path = f"Measurements/Data_Emily/Profil{self.energy}_film.txt"
        film_data = []
        for data in np.loadtxt(film_path):
            film_data.append(data / 100)

        for index, intensity in enumerate(intensity_profiles):
            relative_intensity = intensity / self.profile_grayvalues()[1][index]
            reconstructed_relative_intensity = self.curve_fft(relative_intensity)
            median_index = self.central_axis(reconstructed_relative_intensity)
            off_ax_position_pix = np.linspace(- median_index, len(reconstructed_relative_intensity) - median_index, len(reconstructed_relative_intensity))
            off_ax_position_cm = off_ax_position_pix * self.pixel_converter[0] /10
            film_ax_position_cm = np.linspace(-2.85, 3.15, len(film_data))

            # Create and customize the plot
            fig, ax = plt.subplots()
            palette = sns.color_palette("colorblind")
            ax.plot(off_ax_position_cm, reconstructed_relative_intensity, color=palette[2], linewidth="0.9", label="Raw Cherenkov")
            ax.plot(film_ax_position_cm, film_data, color=palette[4], linewidth="0.9", label="Radiochromic Film")
            ax.legend()
            ax.minorticks_on()
            ax.tick_params(top=True, right=True, axis="both", which="both", direction='in')
            ax.set_ylabel("Relative dose [-]", fontsize=16)
            ax.set_xlabel("Off axis distance [cm]", fontsize=16)
            ax.set_xlim(-2.5, 2.5)

            numbers = "".join(filter(str.isdigit, self.energy))
            text = "".join(filter(str.isalpha, self.energy))
            ax.text(x=1.6, y=0.9, s="{0} {1}".format(numbers, text), fontsize=14)

            # Save the plot as an image
            plt.savefig(os.path.join(directory, "with_ref"), bbox_inches="tight", dpi=600)
            plt.close(fig)

    def calculate_curve_difference(self):
        # Get intensity profiles
        intensity_profiles, _, _, _, _ = self.profile_grayvalues()
        film_path = f"Measurements/Data_Emily/Profil{self.energy}_film.txt"
        film_data = np.loadtxt(film_path) / 100  # Normalize film data

        for intensity in intensity_profiles:
            relative_intensity = intensity / np.max(intensity)
            reconstructed_relative_intensity = self.curve_fft(relative_intensity)
            median_index = self.central_axis(reconstructed_relative_intensity)
            off_ax_position_pix = np.linspace(-median_index, len(reconstructed_relative_intensity) - median_index, len(reconstructed_relative_intensity))
            off_ax_position_cm = off_ax_position_pix * self.pixel_converter[0] / 10
            film_ax_position_cm = np.linspace(-2.85, 3.15, len(film_data))

            # Restrict the data to the interval from -2.5 cm to 2.5 cm
            mask = (off_ax_position_cm >= -2.5) & (off_ax_position_cm <= 2.5)
            off_ax_position_cm = off_ax_position_cm[mask]
            reconstructed_relative_intensity = reconstructed_relative_intensity[mask]

            # Create a mask for film_ax_position_cm and film_data based on the length of position_cm
            film_mask = (film_ax_position_cm >= -2.5) & (film_ax_position_cm <= 2.5)
            film_ax_position_cm = film_ax_position_cm[film_mask]
            film_data = film_data[film_mask]

            # Interpolate the reconstructed curve to match the film data
            interpolated_reconstructed_curve = np.interp(film_ax_position_cm, off_ax_position_cm, reconstructed_relative_intensity)

            # Calculate the difference between the film data and the interpolated curve
            curve_difference = np.abs(film_data - interpolated_reconstructed_curve)

            # Calculate the average difference for this curve
            average_difference = np.mean(curve_difference)
            standard_deviation = np.std(curve_difference)

        return average_difference, standard_deviation

CHECKERBOARD = (7, 10)
DIAGONALE = 25
date = "2023-07-10"
energy = "6MV"
path = f"Measurements/{date}"
analyse = Analysis(CHECKERBOARD, DIAGONALE, path, energy)
difference = analyse.calculate_curve_difference()
print(f"Curve Average Difference: {difference[0]:.3e} ± {difference[1]:.3e}")
analyse.plot_profile()