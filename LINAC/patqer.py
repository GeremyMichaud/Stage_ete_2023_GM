from images_converter import Converter
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob

class Patqer:
    def __init__(self, checkerboard, diagonal_square_size, path, energy, x_offset, half_interval_size):
        calibration_images = glob.glob(f"{path}/Calibration/*")

        self.path = path
        self.energy = energy
        self.x_offset = x_offset
        self.half_interval_size = half_interval_size

        # Initialize pixel-to-mm converter using the provided calibration images
        self.pixel_converter = Converter(calibration_images, checkerboard, diagonal_square_size).convert_pixel2mm(
            calib_image_index=0)

    def vertical_grayvalue(self, image_name):
        image_path = os.path.join(self.path, "Patqer", self.energy, image_name)
        img = plt.imread(image_path)
        img *= 90

        width = img.shape[1]
        central_row = width // 2

        start_line = central_row - half_interval_size - x_offset
        end_line = central_row + half_interval_size - x_offset

        img_centered = []
        for row in img:
            img_centered.append(row[start_line:end_line])

        angles = np.mean(img_centered, axis=1)

        return angles

    def plot_angle(self, image_name):
        angles = self.vertical_grayvalue(image_name)
        depth_pix = np.linspace(0, len(angles), len(angles))
        depth_cm = depth_pix * self.pixel_converter[0] /10

        fig, ax = plt.subplots()
        palette = sns.color_palette("colorblind")
        ax.plot(depth_cm, angles, color=palette[2], linewidth="1", label="Raw Cherenkov")
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
x_offset = 20
half_interval_size = 50
patqer = Patqer(CHECKERBOARD, DIAGONALE, path, energy, x_offset, half_interval_size)


patqer.plot_angle("AoLP.png")