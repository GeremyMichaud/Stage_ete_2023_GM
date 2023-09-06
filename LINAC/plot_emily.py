import matplotlib.pyplot as plt
import numpy as np


film_path = f"Measurements/Data_Emily/RP18MV_film.txt"
film_data = []
for data in np.loadtxt(film_path):
    film_data.append(data)

raw_path = f"Measurements/Data_Emily/RP18MV_raw.txt"
raw_data = []
for data in np.loadtxt(raw_path):
    raw_data.append(data/100)

pola_corr_path = f"Measurements/Data_Emily/RP18MV_pola_corr.txt"
pola_corr_data = []
for data in np.loadtxt(pola_corr_path):
    pola_corr_data.append(data/100)

target_position_cm = 3.0
max_film_index = np.argmax(film_data)
film_data_per_cm = max_film_index / target_position_cm
film_total_cm = len(film_data) / film_data_per_cm
film_position_cm = np.linspace(0, film_total_cm, len(film_data))

plt.plot(film_position_cm, pola_corr_data, color="green", linestyle="dotted", label="Polarized$_{corr}$")
plt.plot(film_position_cm, raw_data, color="blue", linestyle="dashed", label="Raw Cherenkov")
plt.plot(film_position_cm, film_data, color="black", linestyle="solid", label="Film")
plt.tick_params(top=True, right=True, axis="both", which="both", direction='in')
plt.xlim(0,16)
plt.legend()
plt.show()


film_path = f"Measurements/Data_Emily/RP6MV_film.txt"
film_data = []
for data in np.loadtxt(film_path):
    film_data.append(data)

raw_path = f"Measurements/Data_Emily/RP6MV_raw.txt"
raw_data = []
for data in np.loadtxt(raw_path):
    raw_data.append(data/100)

pola_corr_path = f"Measurements/Data_Emily/RP6MV_pola_corr.txt"
pola_corr_data = []
for data in np.loadtxt(pola_corr_path):
    pola_corr_data.append(data/100)

target_position_cm = 1.5
max_film_index = np.argmax(film_data)
film_data_per_cm = max_film_index / target_position_cm
film_total_cm = len(film_data) / film_data_per_cm
film_position_cm = np.linspace(0, film_total_cm, len(film_data))

plt.plot(film_position_cm, pola_corr_data, color="green", linestyle="dotted", label="Polarized$_{corr}$")
plt.plot(film_position_cm, raw_data, color="blue", linestyle="dashed", label="Raw Cherenkov")
plt.plot(film_position_cm, film_data, color="black", linestyle="solid", label="Film")
plt.tick_params(top=True, right=True, axis="both", which="both", direction='in')
plt.xlim(0,16)
plt.legend()
plt.show()