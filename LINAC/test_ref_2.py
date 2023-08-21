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
        intensity_list = []
        max_value_list =[]
        max_index_list = []

        for improved_image in self.improved_images:
            img = plt.imread(improved_image)

            # Calculate the mean intensity along the vertical axis
            intensity = np.mean(img, axis=1)
            intensity_list.append(list(intensity))
            max_index = np.argmax(intensity)
            max_value_list.append(intensity[max_index])
            max_index_list.append(max_index)

        return intensity_list, max_value_list, max_index_list

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

        film_path = f"Measurements/Data_Emily/RP{self.energy}_film.txt"
        film_data = []
        for data in np.loadtxt(film_path):
            film_data.append(data)

        for index, intensity in enumerate(intensity_profiles):
            relative_intensity = intensity / self.pdd_grayvalues()[1][index]
            reconstructed_data = self.curve_fft(relative_intensity)
            position_pix = np.linspace(-160, len(reconstructed_data), len(reconstructed_data))
            position_cm = position_pix * self.pixel_converter[0] /10
            film_ax_position_cm = np.linspace(-0.55, 20.5, len(film_data))

            # Create and customize the plot
            fig, ax = plt.subplots()
            palette = sns.color_palette("colorblind")
            ax.plot(position_cm, reconstructed_data, color=palette[0], linewidth="0.7", label="Camera measurement")
            ax.plot(film_ax_position_cm, film_data, color=palette[3], linewidth="0.7", label="Film reference")
            ax.legend(loc="lower center")
            ax.minorticks_on()
            ax.tick_params(top=True, right=True, axis="both", which="both", direction='in')
            ax.set_ylabel("Percentage depth dose [-]", fontsize=16)
            ax.set_xlabel("Depth [cm]", fontsize=16)
            ax.set_xlim(0, 18)

            numbers = "".join(filter(str.isdigit, self.energy))
            text = "".join(filter(str.isalpha, self.energy))
            ax.text(x=15, y=0.9, s="{0} {1}".format(numbers, text), fontsize=14)

            # Save the plot as an image
            plt.savefig(os.path.join(directory, "with_ref"), bbox_inches="tight", dpi=600)
            plt.close(fig)

    def calculate_curve_difference(self):
        # Get intensity profiles
        intensity_profiles = self.pdd_grayvalues()[0]
        film_path = f"Measurements/Data_Emily/RP{self.energy}_film.txt"
        film_data = np.loadtxt(film_path)  # Normalize film data

        for index, intensity in enumerate(intensity_profiles):
            relative_intensity = intensity / self.pdd_grayvalues()[1][index]
            reconstructed_data = self.curve_fft(relative_intensity)
            position_pix = np.linspace(-160, len(reconstructed_data), len(reconstructed_data))
            position_cm = position_pix * self.pixel_converter[0] /10
            film_ax_position_cm = np.linspace(-0.55, 20.5, len(film_data))

            # Restrict the data to the interval from 0 cm to 18 cm
            mask = (position_cm >= 0) & (position_cm <= 18)
            position_cm = position_cm[mask]
            reconstructed_data = reconstructed_data[mask]

            # Create a mask for film_ax_position_cm and film_data based on the length of position_cm
            film_mask = (film_ax_position_cm >= 0) & (film_ax_position_cm <= 18)
            film_ax_position_cm = film_ax_position_cm[film_mask]
            film_data = film_data[film_mask]

            # Interpolate the reconstructed curve to match the film data
            interpolated_reconstructed_curve = np.interp(film_ax_position_cm, position_cm, reconstructed_data)

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
difference = analyse.calculate_curve_difference()
print(f"Curve Average Difference: {difference[0]:.3e} ± {difference[1]:.3e}")
#analyse.plot_pdd()