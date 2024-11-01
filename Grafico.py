import matplotlib
matplotlib.use('TkAgg')  # Configurar o backend para TkAgg
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid, quad
import scipy.optimize as optimize
from scipy.signal import butter,filtfilt

AttR = []
AttP = []
AttY = []

PosX = []
PosY = []
PosZ = []

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def read_data(file_path):
    with open(file_path, "r") as file:
        return [float(line.strip()) for line in file.readlines()]
    
# Read position data
PosX = read_data("Dados/PosX.txt")
PosY = read_data("Dados/PosY.txt")
PosZ = read_data("Dados/PosZ.txt")

# Read attendance data (if needed)
AttR = read_data("Dados/AttR.txt")
AttP = read_data("Dados/AttP.txt")
AttY = read_data("Dados/AttY.txt")

# Filter requirements.
T = 2.0         # Sample Period
fs = 100.0       # sample rate, Hz
cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

fc = 10  # Frequência de corte
# Filter the data, and plot both the original and filtered signals.
PosXF = butter_lowpass_filter(PosX, cutoff, fs, order)
PosYF = butter_lowpass_filter(PosY, cutoff, fs, order)
PosZF = butter_lowpass_filter(PosZ, cutoff, fs, order)

tempo = np.linspace(0, len(PosX), len(PosX))  # Vetor de tempo

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(tempo, PosXF, label='X')
plt.xlabel('Pontos')
plt.ylabel('Posição')
# plt.title('Posição ao ')
plt.legend()
plt.grid()
#
plt.subplot(1, 3, 2)
plt.plot(tempo, PosYF, label='Y')
plt.xlabel('Pontos ')
plt.ylabel('Posição')
# plt.title('Posição')
plt.legend()
plt.grid()
#
plt.subplot(1, 3, 3)
plt.plot(tempo, PosZF, label='Z')
plt.xlabel('Pontos')
plt.ylabel('Posição')
# plt.title('Posição')
plt.legend()
plt.grid()
#
plt.figure(figsize=(12, 4))

plt.subplot()
plt.plot(PosXF, PosYF, label='Plano xy')
plt.xlabel('x')
plt.ylabel('y')
# plt.title('Posição ao ')
plt.legend()
plt.grid()
plt.show()