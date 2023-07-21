from images_converter import Converter
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import sys

class Profile:
    def __init__(self, checkerboard, diagonal_square_size, path, energy, calib_image_index=0):
        self.improved_images = glob.glob(f"{path}/Improved_Data/{energy}/*")
        calibration_images = glob.glob(f"{path}/Calibration/*")

        self.path = path
        self.energy = energy
        self.folder_path = os.path.join(path, energy)

        self.pixel_converter = Converter(calibration_images, checkerboard, diagonal_square_size).convert_pixel2mm(
            calib_image_index)

    def get_file_names(self):
        """Récupère les noms de fichiers des images.

        Returns:
            list: Liste des noms de fichiers des images sans extension.
        """
        try:
            extension = ".fit"
            file_names = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
            file_names_without_extension = [f[:-len(extension)] if f.endswith(extension) else f for f in file_names]
            return file_names_without_extension
        except OSError as e:
            print(f"Error accessing folder: {e}")
            return []

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

    def curve_fft(self, x_data_graph):
        # Calcul de la transformée de Fourier discrète (DFT)
        coeff_dft = np.fft.rfft(x_data_graph)
        # Calcul du seuil pour conserver les premiers 10% des coefficients
        dft_thresholded_10 = coeff_dft.copy()
        ten_percent = int(0.1*len(dft_thresholded_10))
        # Application du seuil pour ramener à zéro les 90% restants
        dft_thresholded_10[ten_percent:] = 0
        reconstructed_data = np.fft.irfft(dft_thresholded_10)
        return reconstructed_data

    def central_axis(self):
        central_axis = []
        for improved_image in self.improved_images:
            img = plt.imread(improved_image)
            central_axis.append(img[0].size // 2)
        print(central_axis)
        return central_axis

    def plot_grayvalue_profile(self):
        intensity_profiles = self.grayvalues()[0]
        directory = f"{self.path}/Profile/{self.energy}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        for index, intensity in enumerate(intensity_profiles):
            relative_intensity = intensity / self.grayvalues()[1][index]
            reconstructed_relative_intensity = self.curve_fft(relative_intensity)
            middle = np.argmin(abs(np.gradient(reconstructed_relative_intensity, 10)))
            off_ax_position_pix = np.linspace(-middle, len(reconstructed_relative_intensity) - middle, len(reconstructed_relative_intensity))
            off_ax_position_cm = off_ax_position_pix * self.pixel_converter[0] /10
            fig, ax = plt.subplots()
            palette = sns.color_palette("colorblind")
            ax.plot(off_ax_position_cm, reconstructed_relative_intensity, color=palette[2], linewidth="0.5")
            ax.minorticks_on()
            ax.tick_params(top=True, right=True, axis="both", which="both", direction='in')
            ax.set_ylabel("Relative dose [-]", fontsize=16)
            ax.set_xlabel("Off axis distance [cm]", fontsize=16)
            ax.set_xlim(-5, 5)
            numbers = "".join(filter(str.isdigit, self.energy))
            text = "".join(filter(str.isalpha, self.energy))
            ax.text(x=3, y=0.9, s="{0} {1}".format(numbers, text), fontsize=14)
            plt.savefig("{0}/{1}.png".format(directory, self.get_file_names()[index]), bbox_inches="tight", dpi=300)
            plt.close(fig)

date = "2023-06-27"
energy = "6MV"
path = f"Measurements/{date}"
checkerboard = (7, 10)
diagonal = 25

profil = Profile(checkerboard, diagonal ,path, energy)
profil.plot_grayvalue_profile()
