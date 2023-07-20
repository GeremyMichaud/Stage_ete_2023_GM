from images_converter import Converter
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2 as cv


class Profile:
    def __init__(self, checkerboard, diagonal_square_size, path, energy):
        self.improved_images = glob.glob(f"{path}/Improved_Data/{energy}/*")
        calibration_images = glob.glob(f"{path}/Calibration/*")

        self.path = path
        self.energy = energy

        pixel_converter = Converter(self.improved_images, checkerboard, diagonal_square_size)
        self.calibration_images = Converter(calibration_images, checkerboard, diagonal_square_size).convert_fits2png()

    def grayvalues(self):
        intensity_list = []
        max_value_list =[]
        max_index_list = []

        for improved_image in self.improved_images:
            img = plt.imread(improved_image)
            intensity = np.mean(img, axis=0)
            intensity_list.append(list(intensity))
            max_index = np.argmax(intensity)
            max_value_list.append(intensity[max_index])
            max_index_list.append(max_index)

        return intensity_list, max_value_list, max_index_list

    def central_axis(self):
        central_axis = []
        for improved_image in self.improved_images:
            img = plt.imread(improved_image)
            central_axis.append(img[0].size // 2)
        return central_axis


    def plot_grayvalue_profile(self):
        intensity_profiles = self.grayvalues()[0]
        directory = f"{self.path}/Profile/{self.energy}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        for index, intensity in enumerate(intensity_profiles):
            relative_intensity = intensity / self.grayvalues()[1][index]
            off_ax_position = np.linspace(-len(relative_intensity)/2, len(relative_intensity)/2, len(relative_intensity))
            fig, ax = plt.subplots()
            ax.plot(off_ax_position, relative_intensity, color="black", linewidth="0.5")
            ax.minorticks_on()
            ax.tick_params(top=True, right=True, axis="both", which="both", direction='in')
            ax.set_ylabel("Relative dose [-]", fontsize=16)
            ax.set_xlabel("Off axis ditance [cm]", fontsize=16)
            plt.savefig(f"{directory}/{index}.png", bbox_inches="tight", dpi=300)
            plt.close(fig)

date = "2023-06-27"
energy = "6MV"
path = f"Measurements/{date}"
checkerboard = (7, 10)
diagonal = 25

profil = Profile(checkerboard, diagonal ,path, energy)
profil.central_axis()
profil.plot_grayvalue_profile()
