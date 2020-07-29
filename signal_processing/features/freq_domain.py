import numpy as np
import sys

sys.path.append("C:/Users/KyoDante/Desktop/Torch_Signal_package/")

from signal_processing.transformation.fft import fft_

# frequency amplitude
def freq_amp(signal, sampling_rate):
    f = np.arange(0 , sampling_rate , sampling_rate/len(signal))
    y = fft_(signal) * 2/len(signal)
    return np.abs(y[0:len(f)//2])


if __name__ == "__main__":
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

    print(freq_amp(y, fs))