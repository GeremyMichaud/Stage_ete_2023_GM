import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import os

def load_data(filename, scale=1):
    data = np.loadtxt(filename)
    return [x * scale for x in data]

def plot_photon_emily(energy):
    general_path = os.path.join("Measurements", "Data_Emily")
    energy_number = "".join(filter(str.isdigit, energy))
    energy_text = "".join(filter(str.isalpha, energy))

    film_data = load_data(os.path.join(general_path, f"RP{energy}_film.txt"), scale=100)
    raw_data = load_data(os.path.join(general_path, f"RP{energy}_raw.txt"))
    pola_corr_data = load_data(os.path.join(general_path, f"RP{energy}_pola_corr.txt"))

    target_position_cm = 1.5 if energy == "6MV" else 3.0

    max_film_index = np.argmax(film_data)
    film_data_per_cm = max_film_index / target_position_cm
    film_total_cm = len(film_data) / film_data_per_cm
    film_position_cm = np.linspace(0, film_total_cm, len(film_data))

    plt.plot(film_position_cm, film_data, 'ko-', markersize=2, linewidth=1.5, label="Film")
    plt.plot(film_position_cm, raw_data, 'b--', linewidth=1.5, label="Raw Cherenkov")
    plt.plot(film_position_cm, pola_corr_data, 'g:', linewidth=1.5, label="Polarized$_{corr}$")

    plt.tick_params(top=True, right=True, axis="both", which="major", direction='in', labelsize=14)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2.5))
    plt.axvline(x=target_position_cm, color="black", linestyle=":", linewidth=1.2)
    plt.xlim(0, 17)
    plt.ylabel("Relative dose [%]", fontsize=16)
    plt.xlabel("Depth [cm]", fontsize=16)
    plt.text(x=14, y=90, s=f"{energy_number} {energy_text}", fontsize=14)
    plt.legend(loc="lower left", fontsize=16)
    plt.show()

def plot_electron_emily(energy):
    general_path = os.path.join("Measurements", "Data_Emily")
    energy_number = "".join(filter(str.isdigit, energy))
    energy_text = "".join(filter(str.isalpha, energy))

    film_data = load_data(os.path.join(general_path, f"RP{energy}_film.txt"))
    raw_data = load_data(os.path.join(general_path, f"RP{energy}_raw.txt"))
    pola_corr_data = load_data(os.path.join(general_path, f"RP{energy}_pola_corr.txt"))

    plt.plot(film_data, 'ko-', markersize=2, linewidth=1.5, label="Film")
    plt.plot(raw_data, 'b--', linewidth=1.5, label="Raw Cherenkov")
    plt.plot(pola_corr_data, 'g:', linewidth=1.5, label="Polarized$_{corr}$")

    plt.ylabel("Relative dose [%]", fontsize=16)
    plt.xlabel("Depth [count]", fontsize=16)
    plt.legend(loc="lower left", fontsize=16)
    plt.show()

#plot_photon_emily("6MV")
#plot_photon_emily("18MV")
#plot_electron_emily("6MeV")
#plot_electron_emily("18MeV")