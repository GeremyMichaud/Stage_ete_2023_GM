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

    def profile_grayvalues(self):
        """Calculate intensity profiles along a central horizontal interval.

        Returns:
            tuple: Lists of intensity profiles, maximum values, maximum indices, start and end lines.
        """
        interval_size = 30
        y_offset = 200
        intensity_list = []
        max_value_list =[]
        max_index_list = []

        for improved_image in self.improved_images:
            img = plt.imread(improved_image)
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

    def plot_interval(self):
        """Plot intervals of interest on images.
        """
        directory = os.path.join(self.path, "Interval", self.energy)
        os.makedirs(directory, exist_ok=True)

        for count, image in enumerate(self.improved_images):
            image_8bit = cv.imread(image)
            # Highlight the interval on the image using green color
            image_8bit[self.grayvalues()[3], :] = [0, 255, 0]
            image_8bit[self.grayvalues()[4], :] = [0, 255, 0]

            # Save the modified image with interval markers
            cv.imwrite(os.path.join(directory, self.get_file_names()[count]), image_8bit)

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

    def plot_profile(self, plot_interval=False):
        """Plot intensity profiles along the off-axis direction.

        Args:
            plot_interval (bool, optional):  Whether to plot interval markers.. Defaults to False.
        """
        intensity_profiles = self.profile_grayvalues()[0]
        directory = os.path.join(self.path, "Profile", self.energy)
        os.makedirs(directory, exist_ok=True)

        #film_path = f"Measurements/Data_Emily/Profil{self.energy}_film.txt"
        #film_data = []
        #for data in np.loadtxt(film_path):
        #    film_data.append(data / 100)

        if plot_interval:
                self.plot_interval()

        for index, intensity in enumerate(intensity_profiles):
            relative_intensity = intensity / self.profile_grayvalues()[1][index]
            reconstructed_relative_intensity = self.curve_fft(relative_intensity)
            median_index = self.central_axis(reconstructed_relative_intensity)
            off_ax_position_pix = np.linspace(- median_index, len(reconstructed_relative_intensity) - median_index, len(reconstructed_relative_intensity))
            off_ax_position_cm = off_ax_position_pix * self.pixel_converter[0] /10
            #film_ax_position_cm = np.linspace(-2.85, 3.15, len(film_data))

            # Create and customize the plot
            fig, ax = plt.subplots()
            palette = sns.color_palette("colorblind")
            ax.plot(off_ax_position_cm, reconstructed_relative_intensity, color=palette[2], linewidth="0.7")
            #ax.plot(film_ax_position_cm, film_data, color=palette[0], linewidth="0.7")
            ax.minorticks_on()
            ax.tick_params(top=True, right=True, axis="both", which="both", direction='in')
            ax.set_ylabel("Relative dose [-]", fontsize=16)
            ax.set_xlabel("Off axis distance [cm]", fontsize=16)
            ax.set_xlim(-3, 3)

            numbers = "".join(filter(str.isdigit, self.energy))
            text = "".join(filter(str.isalpha, self.energy))
            ax.text(x=2, y=0.9, s="{0} {1}".format(numbers, text), fontsize=14)

            # Save the plot as an image
            plt.savefig(os.path.join(directory, self.get_file_names()[index]), bbox_inches="tight", dpi=600)
            plt.close(fig)

    def plot_pdd(self):
        """Plot Percentage Depth Dose (PDD) profiles.
        """
        intensity_profiles = self.pdd_grayvalues()[0]
        directory = os.path.join(self.path, "PDD", self.energy)
        os.makedirs(directory, exist_ok=True)

        for index, intensity in enumerate(intensity_profiles):
            relative_intensity = intensity / self.pdd_grayvalues()[1][index]
            reconstructed_data = self.curve_fft(relative_intensity)
            position_pix = np.linspace(0, len(reconstructed_data), len(reconstructed_data))
            position_cm = position_pix * self.pixel_converter[0] /10

            # Create and customize the plot
            fig, ax = plt.subplots()
            palette = sns.color_palette("colorblind")
            ax.plot(position_cm, reconstructed_data, color=palette[0], linewidth="0.7")
            ax.minorticks_on()
            ax.tick_params(top=True, right=True, axis="both", which="both", direction='in')
            ax.set_ylabel("Percentage depth dose [-]", fontsize=16)
            ax.set_xlabel("Depth [cm]", fontsize=16)

            numbers = "".join(filter(str.isdigit, self.energy))
            text = "".join(filter(str.isalpha, self.energy))
            ax.text(x=20, y=0.9, s="{0} {1}".format(numbers, text), fontsize=14)

            # Save the plot as an image
            plt.savefig(os.path.join(directory, self.get_file_names()[index]), bbox_inches="tight", dpi=600)
            plt.close(fig)
