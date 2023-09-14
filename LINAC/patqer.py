from images_converter import Converter
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv
import glob

class Patqer:
    def __init__(self, checkerboard, diagonal_square_size, path, energy, half_interval_size, image_name, x_offset=0):
        calibration_images = glob.glob(f"{path}/Calibration/*")

        self.path = path
        self.energy = energy
        self.x_offset = x_offset
        self.half_interval_size = half_interval_size
        self.image_name = image_name

        # Initialize pixel-to-mm converter using the provided calibration images
        self.pixel_converter = Converter(calibration_images, checkerboard, diagonal_square_size).convert_pixel2mm(
            calib_image_index=0)

    def vertical_grayvalue(self):
        image_path = os.path.join(self.path, "Patqer", self.energy, self.image_name)
        img = plt.imread(image_path)
        img *= 90

        width = img.shape[1]
        central_row = width // 2

        start_line = central_row - half_interval_size - self.x_offset
        end_line = central_row + half_interval_size - self.x_offset

        img_centered = []
        for row in img:
            img_centered.append(row[start_line:end_line])

        angles = np.mean(img_centered, axis=1)

        return angles

    def curve_fft(self, x_data_graph, threshold=10):
        """Perform Fourier transform and return the reconstructed data after thresholding.

        Args:
            x_data_graph (array-like): Input data for Fourier transform.
            threshold (int, float): Poucentage of threshold that will be applied. Default to 10%.

        Returns:
            array-like: Reconstructed data after thresholding.
        """
        # Calculate discrete Fourier transform (DFT)
        coeff_dft = np.fft.rfft(x_data_graph)
        # Calculate threshold to keep the first [threshold]% of coefficients
        dft_thresholded = coeff_dft.copy()
        threshold_percent = int(len(dft_thresholded) / threshold)
        # Apply threshold to zero out the remaining 90% coefficients
        dft_thresholded[threshold_percent:] = 0
        # Perform inverse Fourier transform to reconstruct the data
        reconstructed_data = np.fft.irfft(dft_thresholded)
        return reconstructed_data

    def plot_vertical_interval(self):
        image_path = os.path.join(self.path, "Patqer", self.energy, self.image_name)
        img = cv.imread(image_path)
        width = img.shape[1]
        central_row = width // 2
        start_line = central_row - self.half_interval_size - self.x_offset
        end_line = central_row + self.half_interval_size - self.x_offset
        cv.line(img, (start_line, 0), (start_line, img.shape[0]), (0,255, 0), thickness=2)
        cv.line(img, (end_line, 0), (end_line, img.shape[0]), (0,255, 0), thickness=2)
        cv.imshow('IGL.png', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def plot_angle(self):
        angles = self.vertical_grayvalue()
        angles_fft = self.curve_fft(angles, threshold=10)
        depth_pix = np.linspace(0, len(angles_fft), len(angles_fft))
        depth_cm = depth_pix * self.pixel_converter[0] /10

        _, ax = plt.subplots()
        palette = sns.color_palette("colorblind")
        ax.plot(depth_cm, angles_fft, color=palette[2], linewidth="1", label="Raw Cherenkov")
        ax.legend(fontsize=16)
        ax.minorticks_on()
        ax.tick_params(top=True, right=True, axis="both", which="both", direction='in')
        ax.set_ylabel("Agnle of linear polarization [°]", fontsize=16)
        ax.set_xlabel("Depth [cm]", fontsize=16)

        numbers = "".join(filter(str.isdigit, self.energy))
        text = "".join(filter(str.isalpha, self.energy))
        ax.text(x=13, y=60, s="{0} {1}".format(numbers, text), fontsize=14)

        plt.show()


CHECKERBOARD = (7, 10)
DIAGONALE = 25
date = "2023-07-24"
energy = "6MV"
path = os.path.join("Measurements", date)
half_interval_size = 10
image_name= "DoLP.png"
patqer = Patqer(CHECKERBOARD, DIAGONALE, path, energy, half_interval_size, image_name)

patqer.plot_vertical_interval()
patqer.plot_angle()