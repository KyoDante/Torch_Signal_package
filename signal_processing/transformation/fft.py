import scipy.signal as sig
import scipy.fft as fft

### ref: https://mp.weixin.qq.com/s/flmcvRxKx5DkaufoHn0PMw
### 描述和解释信号的常用量
# x:  采样的数据；
# n=length(x): 样本数量；
# fs: 采样频率(每单位时间或空间的样本数)(单位常用:赫兹Hz);
# dt=1/fs: 每样本的时间或空间增量(如果是时间上的增量，则又称：采样间隔或采样步长，单位常用:s);
# t=(0:n-1)/fs; 数据的时间或空间范围;
# y=fft(x): 数据的离散傅里叶变换(DFT);
# abs(y): DFT的振幅;
# (abs(y).^2)/n: DFT的幂;
# fs/n: 频率增量;
# f=(0:n-1) * (fs/n): 频率范围;
# fs/2: Nyquist频率(频率范围的中点);
# 最后结果乘 2/n，则获得前面的常数系数的值。

def fft_(time_series):
    return fft.fft(time_series)

# short-time fourier transform
def stft(time_series, sampling_rate):
    # fs=1.0, window='hann', 
    # nperseg=256, noverlap=None, 
    # nfft=None, detrend=False, 
    # return_onesided=True, boundary='zeros', padded=True
    # detrend?
    f, t, Zxx = sig.stft(time_series, sampling_rate)
    return f, t, Zxx

# spectrogram
def spectrogram(time_series, sampling_rate):
    # window=('tukey',0.25), 
    # nperseg=None, noverlap=None, 
    # nfft=None, detrend='constant', 
    # return_onesided=True, scaling='density', 
    # axis=-1, mode='psd'
    ### 什么是 'psd'? 'welch'?
    f, t, Sxx = sig.spectrogram(time_series, sampling_rate)
    return f, t, Sxx


if __name__ == "__main__":
    import numpy as np
    import pylab as pl
    import math

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
    pl.plot(t,y)
    pl.xlabel('time(s)')
    pl.title("original signal")
    pl.show()

    # 采样点数
    N=len(t)
    # 采样频率
    fs=1048.0
    # 分辨率
    df = fs/(N-1)
    # 构建频率数组
    f = [df*n for n in range(0,N)]
    Y = np.fft.fft(y)*2/N  #*2/N 反映了FFT变换的结果与实际信号幅值之间的关系
    absY = [np.abs(x) for x in Y]      #求傅里叶变换结果的模

    pl.plot(f,absY)
    pl.xlabel('freq(Hz)')
    pl.title("fft")
    pl.show()

    from scipy.fft import fft, fftshift, ifft
    from scipy.fft import fftfreq
    import numpy as np
    import matplotlib.pyplot as plt

    """
    t_s:采样周期
    t_start:起始时间
    t_end:结束时间
    """
    t_s = 0.01
    t_start = 0.5
    t_end = 5
    t = np.arange(t_start, t_end, t_s)

    f0 = 5
    f1 = 20

    # 绘制图表
    plt.figure(figsize=(10, 12))

    # 构建原始信号序列
    y = 1.5*np.sin(2*np.pi*f0*t) + 3*np.sin(2*np.pi*20*t) + np.random.randn(t.size)
    ax=plt.subplot(511)
    ax.set_title('original signal')
    plt.tight_layout()
    plt.plot(y)

    """
    FFT(Fast Fourier Transformation)快速傅里叶变换
    """
    Y = fft(y) * 2 / len(y)
    ax=plt.subplot(512)
    ax.set_title('fft transform')
    plt.plot(np.arange(0, 1/t_s, 1/t_s/len(y)),np.abs(Y))

    """
    Y = fftshift(X) 通过将零频分量移动到数组中心，重新排列傅里叶变换 X。
    """
    shift_Y = fftshift(Y)
    ax=plt.subplot(513)
    ax.set_title('shift fft transform')
    plt.plot(np.abs(shift_Y))

    """
    得到正频率部分
    """
    pos_Y_from_fft = Y[:Y.size//2]
    ax=plt.subplot(514)
    ax.set_title('fft transform')
    plt.tight_layout()
    plt.plot(np.abs(pos_Y_from_fft))

    """
    直接截取 shift fft结果的前半部分
    """
    pos_Y_from_shift = shift_Y[shift_Y.size//2:]
    ax=plt.subplot(515)
    ax.set_title('shift fft cut')
    plt.plot(np.abs(pos_Y_from_shift))
    plt.show()