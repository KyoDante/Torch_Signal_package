import sys
sys.path.append("C:/Users/KyoDante/Desktop/Torch_Signal_package/")

from signal_processing.transformation.fft import fft_
import numpy as np

# energy
def band_energy(signal_data, sampling_rate, f0, f1):
    f = np.arange(0,sampling_rate,sampling_rate/len(signal_data))
    fft_energy = np.abs(fft_(signal_data) * 2/len(signal_data))**2
    band_idx = np.logical_and(f<f1,f0<f)
    return np.sum(fft_energy[band_idx])

if __name__ == "__main__":
    # 采样频率
    fs=1048
    # 采样步长
    t = [x/1048.0 for x in range(1048)]
    """
    设计的采样值
    假设信号y由4个周期信号叠加所得,如下所示
    """
    y = [ 3.0 * np.cos(2.0 * np.pi * 50 * t0 - np.pi * 30/180)
        + 1.5 * np.cos(2.0 * np.pi * 75 * t0 + np.pi * 90/180)
        +  1.0 * np.cos(2.0 * np.pi * 150 * t0 + np.pi * 120/180)
        +  2.0 * np.cos(2.0 * np.pi * 220 * t0 + np.pi * 30/180)
        for t0 in t ]
    print(band_energy(y, fs, 40, 80))