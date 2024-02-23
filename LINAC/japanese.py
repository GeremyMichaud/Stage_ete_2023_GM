import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d, savgol_filter
import os
import glob
from images_converter import Converter
from camera_calibrator import CameraCalibrator
from imageio import imread


class Japanese:
    def __init__(self, checkerboard, diagonal_square_size, path, energy):
        """Classe pour améliorer les données d'images en enlevant l'arrière-plan et en redressant les images.

        Args:
            checkerboard (tuple): Un tuple contenant le nombre de coins du damier dans les directions x et y.
            diagonal_square_size (float): Taille de la diagonale d'un carré du damier en millimètres.
            path (str): Chemin du répertoire contenant les images.
            energy (str): Nom de l'énergie des images.
        """
        raw_images = glob.glob(f"{path}/{energy}/*")
        self.backgrounds_path = glob.glob(f"{path}/Background/*")
        calibration_images = glob.glob(f"{path}/Calibration/*")

        self.calibrator = CameraCalibrator(checkerboard, diagonal_square_size, calibration_images)
        self.raw_images = Converter(raw_images, checkerboard, diagonal_square_size).convert_fits2png()
        self.backgrounds = Converter(self.backgrounds_path, checkerboard, diagonal_square_size).convert_fits2png()
        self.pixel_converter = Converter(calibration_images, checkerboard, diagonal_square_size).convert_pixel2mm(
            calib_image_index=0)

        self.path = path
        self.energy = energy
        self.undistorted_images = []
        self.cleaned_images = {}

    def get_file_names(self):
        """Récupère les noms de fichiers des images.

        Returns:
            list: Liste des noms de fichiers des images sans extension.
        """
        try:
            folder_path = f"{self.path}/{self.energy}"
            extension = ".fit"
            file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            file_names_without_extension = [f[:-len(extension)] if f.endswith(extension) else f for f in file_names]
            return file_names_without_extension
        except OSError as e:
            print(f"Error accessing folder: {e}")
            return []

    def remove_background(self):
        """Supprime l'arrière-plan des images brutes.

        Raises:
            ValueError: Si les images n'ont pas les mêmes dimensions.

        Returns:
            list: Liste des images sans l'arrière-plan.
        """
        for bg_index, background_path in enumerate(self.backgrounds_path):
            bg = self.backgrounds[bg_index]
            selected_noisy_images = []
            file_name = os.path.split(background_path)[-1]
            energy_string = file_name.split("_")[0]
            angle_string = file_name.split("_")[1]
            if energy_string == self.energy:
                for i, name in enumerate(self.get_file_names()):
                    angle = name.split("_")[0]
                    if angle == angle_string:
                        selected_noisy_images.append(i)
                for noisy_index in selected_noisy_images:
                    noisy_image = self.raw_images[noisy_index]
                    cleaned_image = cv.absdiff(noisy_image, bg)
                    file_name = self.get_file_names()[noisy_index]
                    self.cleaned_images[file_name] = cleaned_image
            else:
                continue

        if len(self.cleaned_images) == 0:
            print("allo")
            matrice_bg = np.dstack((self.backgrounds))
            mean_bg = np.mean(matrice_bg, axis=2).astype(np.uint16)
            for i, noisy_image in enumerate(self.raw_images):
                # Vérifier que les images ont les mêmes dimensions
                if noisy_image.shape != mean_bg.shape:
                    raise ValueError("The images do not have the same dimensions.")
                # Soustraction de l'image de background de l'image avec bruit
                cleaned_image = cv.absdiff(noisy_image, mean_bg)
                file_name = self.get_file_names()[i]
                self.cleaned_images[file_name] = cleaned_image
        return self.cleaned_images

    def radiative_noise_remove_outliers(self, radius=5, threshold=1000):
        """Calculate the temporal median radiative noise for each image prefix.

        Args:
            radius (int): Radius for the median filter.
            threshold (int): Threshold for outlier detection

        Returns:
            dict: A dictionary containing median radiative noise arrays for each image prefix.
        """
        # Remove background from images
        backgroundless = self.remove_background()
        print(backgroundless)

        # Organize filenames by their prefixes
        prefix_dict = {}
        for filename in backgroundless.keys():
            parts = filename.split("_")
            if parts[0] not in prefix_dict:
                prefix_dict[parts[0]] = []
            prefix_dict[parts[0]].append(filename)

        # Remove outliers for each prefix
        noiseless_dict = {}
        for prefix, filenames in prefix_dict.items():
            arrays_to_remove_outliers = backgroundless[filenames[0]]  # Only take the first image of the serie
            # Applique un filtre médian pour calculer la médiane locale
            kernel_size = 2 * radius + 1
            kernel_size = min(kernel_size, min(arrays_to_remove_outliers[0].shape))
            median_images = [medfilt2d(image, kernel_size=kernel_size) for image in arrays_to_remove_outliers]
            # Calcule la différence entre les images originales et les images médianes
            diff_images = [np.abs(original - median) for original, median in zip(arrays_to_remove_outliers, median_images)]
            # Crée un masque en fonction du seuil
            masks = [diff_image > threshold for diff_image in diff_images]
            for i, mask in enumerate(masks):
                arrays_to_remove_outliers[i][mask] = median_images[i][mask]
            noiseless_dict[prefix] = arrays_to_remove_outliers

        return noiseless_dict

    def plot_colormap(self, images, vvmax, colormap_name="inferno"):
        """Plot colormap images for each prefix using the calculated radiative noise.

        Args:
            colormap_name (str, optional): Name of the colormap. Defaults to "viridis".
        """
        # Define directory to save colormap images
        directory = os.path.join(self.path, "Japan_Data", "Colormap", self.energy)
        if not os.path.exists(directory):
                os.makedirs(directory)

        # Generate and save colormap images
        for name, image_data in images.items():
            fig, ax = plt.subplots()
            im = ax.imshow(image_data, cmap=colormap_name, vmin=0, vmax=vvmax)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
            cbar.set_label('Gray Value', fontsize=16)
            ax.tick_params(left=False, bottom=False, labelleft = False ,
                labelbottom = False)
            ax.text(0.03, 0.98, self.energy, transform=ax.transAxes,
                fontsize=12, color='black',
                bbox=dict(facecolor="w", edgecolor='k', boxstyle='round,pad=0.4'))

            # Save the colormap image with transparency
            plt.savefig(os.path.join(directory, name),
                bbox_inches ="tight", dpi=300, transparent=True)
            plt.close(fig)

    def improve_data(self, colormap_max=20000):
        """Améliore les images en supprimant l'arrière-plan et en réduisant le bruit radiatif.
        Les images améliorées sont enregistrées dans un répertoire 'Improved_Data'.
        """
        directory = os.path.join(self.path, "Japan_Data", self.energy)

        if not os.path.exists(directory):
            os.makedirs(directory)

        images = self.radiative_noise_remove_outliers()

        self.plot_colormap(images, vvmax=colormap_max)

        for name, improved_image in images.items():
            cv.imwrite(os.path.join(directory, name + ".png"), improved_image)

    def polarizing_component(self, colormap_max=20000):
        directory = os.path.join(self.path, "Japan_Data", self.energy)

        if not os.path.exists(directory):
            os.makedirs(directory)

        images = self.radiative_noise_remove_outliers()
        parallel, perpendicular = images["0deg"], images["90deg"]

        parallel = parallel.astype(np.int16)
        perpendicular = perpendicular.astype(np.int16)

        polarizing_component_array = np.clip(parallel - perpendicular, 0, None).astype(np.uint16)

        cv.imwrite(os.path.join(directory, "pol_comp.png"), polarizing_component_array)
        polarizing_component = {"pol_comp": polarizing_component_array}
        self.plot_colormap(polarizing_component, vvmax=colormap_max)
        return perpendicular, polarizing_component["pol_comp"]

    def non_polarized(self, colormap_max=20000):
        directory = os.path.join(self.path, "Japan_Data", self.energy)

        if not os.path.exists(directory):
            os.makedirs(directory)

        perpendicular = self.polarizing_component()[0]
        polaring_component = self.polarizing_component()[1]

        nonpolarized_array = np.clip(perpendicular-polaring_component, 0, None).astype(np.uint16)

        cv.imwrite(os.path.join(directory, "non_pol.png"), nonpolarized_array)
        nonpolarized = {"non_pol": nonpolarized_array}
        self.plot_colormap(nonpolarized, vvmax=colormap_max)
        return nonpolarized_array

    def pdd_grayvalues(self, image, width):
        """Calculate intensity profiles along the vertical axis (percentage depth dose).

        Returns:
            tuple: Lists of intensity profiles, maximum values, and maximum indices.
        """
        img = imread(image)
        center_index = img.shape[1] // 2
        img_cut = img[:, center_index - (width // 2):center_index + (width // 2)]

        # Calculate the mean intensity along the vertical axis
        intensity = np.mean(img_cut, axis=1)
        max_index = np.argmax(intensity)
        max_value = intensity[max_index]

        return intensity, max_value, max_index

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
        max_value = np.max(reconstructed_data)
        max_index = np.argmax(reconstructed_data)
        return reconstructed_data, max_value, max_index

    def savgol(self, y_data, wl=21, po=3):
        """Smooth (filter) a signal using a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.

        Args:
            y_data (array-like): The input signal.
            window_length (int): Window length used for the filtering. Default is 21.
            polyorder (int): Polynomial order of the polynomial that is used to fit the
                data. Must be less than window_length/2 - 1. Default is 3.

        Returns:
            array-like: Smoothed version of the input signal.
        """
        y_smooth = savgol_filter(y_data, window_length=wl, polyorder=po, mode="nearest")
        max_value = np.max(y_smooth)
        max_index = np.argmax(y_smooth)
        return y_smooth, max_value, max_index

    def plot_pdd(self):
        """Plot Percentage Depth Dose (PDD) profiles.
        """
        im_path = os.path.join(self.path, "Japan_Data", self.energy)
        images = glob.glob(os.path.join(im_path, "*"))
        pix_width = 60

        if self.energy == "6MV":
            pattern = "6X"
        elif self.energy == "6MeV":
            pattern = "E06"
        elif self.energy == "12MeV":
            pattern = "E12"
        else:
            raise FileNotFoundError(f"There is no reference data for energy {self.energy}.")
        ref_path = glob.glob(os.path.join("Measurements", "Ion_chamber", "PDD", f"{pattern}*"))
        ion_chamber_data = np.loadtxt(ref_path[0], skiprows=1)
        ion_chamber_depth = ion_chamber_data[:,0]
        ion_chamber_dose = ion_chamber_data[:,1]
        ion_chamber_max_dose = np.max(ion_chamber_dose)
        ion_chamber_index_max = np.argmax(ion_chamber_dose)
        ion_chamber_depth_max = ion_chamber_depth[ion_chamber_index_max]
        ion_chamber_relative_dose = ion_chamber_dose / ion_chamber_max_dose

        nonpolarized_path = images[images.index(os.path.join(im_path, "non_pol.png"))]
        polarized_path = images[images.index(os.path.join(im_path, "pol_comp.png"))]
        perpendicular_path = images[images.index(os.path.join(im_path, "90deg.png"))]
        parallel_path = images[images.index(os.path.join(im_path, "0deg.png"))]

        nonpolarized = self.pdd_grayvalues(nonpolarized_path, pix_width)
        nonpolarized_smooth = self.savgol(nonpolarized[0])
        relative_nonpolarized = nonpolarized_smooth[0] / nonpolarized_smooth[1]
        polarized = self.pdd_grayvalues(polarized_path, pix_width)
        polarized_smooth = self.savgol(polarized[0])
        relative_polarized = polarized_smooth[0] / polarized_smooth[1]
        perpendicular = self.pdd_grayvalues(perpendicular_path, pix_width)
        perpendicular_smooth = self.savgol(perpendicular[0])
        relative_perpendicular = perpendicular_smooth[0] / perpendicular_smooth[1]
        parallel = self.pdd_grayvalues(parallel_path, pix_width)
        parallel_smooth = self.savgol(parallel[0])
        relative_parallel = parallel_smooth[0] / parallel_smooth[1]
        
        """relative_nonpolarized_2 = nonpolarized[0] / np.max(nonpolarized[0])
        relative_polarized_2 = polarized[0] / np.max(polarized[0])
        relative_perpendicular_2 = perpendicular[0] / np.max(perpendicular[0])
        relative_parallel_2 = parallel[0] / np.max(parallel[0])"""

        pixel_factor = self.pixel_converter[0]
        pixel_shift = ion_chamber_depth_max / pixel_factor
        start_pix = -nonpolarized_smooth[2] + pixel_shift
        end_pix = len(nonpolarized_smooth[0]) - nonpolarized_smooth[2] + pixel_shift
        position_pix = np.linspace(start_pix, end_pix, len(nonpolarized_smooth[0]))
        position_mm = position_pix * pixel_factor

        directory = os.path.join(self.path, "Japan_Data", "PDD", self.energy)
        os.makedirs(directory, exist_ok=True)

        fig, ax = plt.subplots()
        ax.plot(position_mm, relative_parallel, linestyle="solid", color="blue", label="Parallel")
        ax.plot(position_mm, relative_perpendicular, linestyle="dashed", color="red", label="Perpendicular")
        ax.plot(position_mm, relative_polarized, linestyle="dotted", color="green", label="Polarized")
        ax.plot(position_mm, relative_nonpolarized, linestyle="dashdot", color="darkviolet", label="Non-polarized")
        ax.plot(ion_chamber_depth, ion_chamber_relative_dose, linestyle=(5,(10,3)), color="orange", label="Ionization chamber")
        
        """ax.plot(position_mm, relative_parallel_2, linestyle="solid", color="cyan", label="Parallel_Unsmooth")
        ax.plot(position_mm, relative_perpendicular_2, linestyle="dashed", color="lightcoral", label="Perpendicular_Unsmooth")
        ax.plot(position_mm, relative_polarized_2, linestyle="dotted", color="palegreen", label="Polarized_Unsmooth")
        ax.plot(position_mm, relative_nonpolarized_2, linestyle="dashdot", color="violet", label="Non-polarized_Unsmooth")"""

        """ax.plot(position_mm, parallel_fft[0], linestyle="solid", color="blue", label="Parallel")
        ax.plot(position_mm, perpendicular_fft[0], linestyle="dashed", color="red", label="Perpendicular")
        ax.plot(position_mm, polarized_fft[0], linestyle="dotted", color="green", label="Polarized")
        ax.plot(position_mm, nonpolarized_fft[0], linestyle="dashdot", color="darkviolet", label="Non-polarized")"""

        ax.minorticks_on()
        ax.tick_params(top=True, right=True, axis="both", which="both", direction='in')
        ax.set_ylabel("Relative dose [-]", fontsize=16)
        ax.set_xlabel("Depth [mm]", fontsize=16)
        ax.set_xlim(0, 50)

        numbers = "".join(filter(str.isdigit, self.energy))
        text = "".join(filter(str.isalpha, self.energy))
        ax.text(x=20, y=0.9, s="{0} {1}".format(numbers, text), fontsize=14)
        plt.legend()
        plt.show()
        #plt.close(fig)

    def verif_factor(self):
        im_path = os.path.join(self.path, "Improved_Data", self.energy)
        images = glob.glob(os.path.join(im_path, "*"))
        pix_width = 60
        pix_height = 200

        perpendicular_path = images[images.index(os.path.join(im_path, "90deg.png"))]
        parallel_path = images[images.index(os.path.join(im_path, "0deg.png"))]

        perpendicular = self.pdd_grayvalues(perpendicular_path, pix_width)
        perpendicular_smooth = self.savgol(perpendicular[0])
        parallel = self.pdd_grayvalues(parallel_path, pix_width)
        parallel_smooth = self.savgol(parallel[0])

        middle = len(perpendicular[0]) // 2
        roi_parallel = parallel[0][middle-pix_height//2:middle+pix_height//2]
        roi_perpendicular = perpendicular[0][middle-pix_height//2:middle+pix_height//2]
        factor_roi = roi_parallel / roi_perpendicular
        factor = parallel[1] / perpendicular[1]
        factor_smooth = parallel_smooth[1] / perpendicular_smooth[1]
        print(f"ROI Factor : \t \t {np.mean(factor_roi):.3f} ± {np.std(factor_roi):.3f}")
        print(f"Max Factor : \t \t {factor:.3f}")
        print(f"Smooth  Max Factor : \t {factor_smooth:.3f}")

        directory = os.path.join(self.path, "Japan_Data", "PDD", self.energy)
        os.makedirs(directory, exist_ok=True)

        fig, ax = plt.subplots()
        position_pix = np.linspace(0, len(parallel_smooth[0]), len(parallel_smooth[0]))

        ax.plot(position_pix, parallel_smooth[0], linestyle="solid", color="cyan", label="Parallel_Smooth")
        ax.plot(position_pix, perpendicular_smooth[0], linestyle="dashed", color="lightcoral", label="Perpendicular_Smooth")

        ax.minorticks_on()
        ax.tick_params(top=True, right=True, axis="both", which="both", direction='in')
        ax.set_ylabel("Gray value [-]", fontsize=16)
        ax.set_xlabel("Depth [pixel]", fontsize=16)

        numbers = "".join(filter(str.isdigit, self.energy))
        text = "".join(filter(str.isalpha, self.energy))
        text_height = parallel_smooth[1] - parallel_smooth[1]/6
        ax.text(x=20, y=text_height, s="{0} {1}".format(numbers, text), fontsize=14)
        plt.legend()
        plt.show()
        #plt.close(fig)