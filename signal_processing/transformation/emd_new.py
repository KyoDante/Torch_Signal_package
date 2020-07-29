import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate

def peaks_num(x):
    max_peaks = signal.argrelmax(x)[0]
    min_peaks = signal.argrelmin(x)[0]
    peaks_num = len(max_peaks) + len(min_peaks)
    return peaks_num

def is_monotic(x):
    peak_num = peaks_num(x)
    if peak_num > 0:
        return False
    else:
        return True

def is_imf(x):
    N = np.size(x)
    pass_zero = np.sum(x[0:N-1] * x[1:N] < 0)
    peak_num = peaks_num(x)
    if abs(pass_zero - peak_num) > 1:
        return False
    return True

def get_spline(x):
    N = np.size(x)
    peaks = signal.argrelmax(x)[0]
    peaks = np.hstack(([0], peaks, [N-1]))

    if len(peaks) <= 3:
        t = interpolate.splrep(peaks, y=x[peaks], w=None, xb=None, xe=None, k=len(peaks)-1)
        return interpolate.splev(np.arange(N), t)
    t = interpolate.splrep(peaks, y=x[peaks], k=3)
    return interpolate.splev(np.arange(N), t)

def emd(signal_data):
    imfs = []
    while not is_monotic(signal_data):
        x1 = signal_data
        sd = np.Inf
        while sd > 0.1 or (not is_imf(x1)):
            s1 = get_spline(x1)
            s2 = -get_spline(-x1)
            x2 = x1 - (s1+s2)/2
            sd = np.sum((x1-x2)**2) / np.sum(x1**2)
            x1 = x2
        
        imfs.append(x1)
        signal_data = signal_data - x1
    
    imfs.append(signal_data)
    return imfs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    t = np.arange(0,1, 0.01)
    S = 2*np.sin(2*np.pi*15*t) +4*np.sin(2*np.pi*10*t)*np.sin(2*np.pi*t*0.1)+np.sin(2*np.pi*5*t)
    plt.figure()
    imfs = emd(S)
    length = np.shape(imfs)[0]
    print(length)
    for i in range(1,length+1):
        plt.subplot(length,1,i)
        plt.plot(imfs[i-1])
    plt.tight_layout()
    plt.show()

    # ref: pip install EMD-signal
    # pyemd库
    # 使用pyemd最好
    import PyEMD.EMD as emd
    imfs = emd()(S)
    length = np.shape(imfs)[0]
    print(length)
    for i in range(1,length+1):
        plt.subplot(length,1,i)
        plt.plot(imfs[i-1])
    plt.tight_layout()
    plt.show()