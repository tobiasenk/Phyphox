import numpy as np
import pandas as pd
from utils import vis
import matplotlib.pyplot as plt

"""
Script reads phyphox experiment data and plot them
- time series
- fft
- spctrogram
"""


plt.close('all')

# Paramter
folder = 'experiments'
experiment = 'Test 2022-10-20_14-25-10'
file = str(folder)+'\\'+str(experiment)+'\Accelerometer.csv'

# Choose time range
start = 100
end = 5000

# optional
time_to_meter = False
web_speed = 150/60  # m/s

# axis labels & units
y_label = 'acceleration'
y_unit = '(m/s^2)'
x_label = 'time'
x_unit = '(s)'

# read csv data
first_data_row = 0
data = pd.read_csv(file, delimiter=',', header=0, index_col=None, skiprows=[i for i in range(1, first_data_row)])

# prepare data
x = data[data.columns[0]]
dt = x[1]-x[0]
x = x  # from ms to s
X_AXIS = data[data.columns[1]]
Y_AXIS = data[data.columns[2]]
Z_AXIS = data[data.columns[3]]

if time_to_meter == True:
    x = x * web_speed
    x_unit = '(m)'
    x_label = 'meter'

# data visualization
y = Y_AXIS
FFT = vis.fft(x, y)
STFT = vis.stft(x, y)

# 3 row subplot of single sensor
fig, ax = plt.subplots(3, 1, figsize=(16.1/2.54, 20/2.54))  #
#plt.figure(figsize=(16.1/2.54, 10/2.54))
# TS
ax[0].set_title('Timeseries')
ax[0].plot(x, y, 'k-', linewidth=0.5)
ax[0].set_ylabel(str(y_label)+' '+str(y_unit))
ax[0].set_xlabel(str(y_label)+' '+str(x_unit))
ax[0].set_xlim(x.iloc[0], x.iloc[-1])
ax[0].grid()
# FFT
ax[1].set_title('FFT Magnitude')
ax[1].plot(FFT['x'], FFT['y'], 'k-', linewidth=0.5)
ax[1].set_xlabel('frequency (Hz)')
ax[1].set_ylabel('amplitude (m/s^2)')
ax[1].set_ylim(0, FFT['y'][10:].max())  # ignore DC component
ax[1].grid()
# STFT
pcm = ax[2].pcolormesh(STFT['t_new'], STFT['f'], np.abs(STFT['Sxx']), vmin=0, vmax=0.1, shading='gouraud')
ax[2].set_ylabel('frequency (Hz)')
ax[2].set_xlabel('time (s)')
fig.colorbar(pcm, ax=ax[2], location='bottom')
ax[2].set_title('STFT Magnitude')
fig.tight_layout()
# save as PNG
plt.savefig('plot\experiment.png', bbox_inches='tight', format='png', dpi=300)



print('end')