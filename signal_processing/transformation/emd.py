import scipy.signal as sig
import scipy.interpolate as interpolate
import numpy as np

# main ref: https://mp.weixin.qq.com/s/oOp5ifDVqqiPNQzdTaaSmg
# ref: https://mp.weixin.qq.com/s/ibnry5Dn2uVjcnKqS6hYBw

# Hilbert变换算法要求输入信号只能是线性稳态的。
# 无论是在自然界还是在人类社会中，绝大部分的信号要么是“线性非稳态”，要么是“非线性稳态”，要么干脆是“非线性非稳态”。
# EEG信号正是这样一类“非线性非稳态”的信号。
# Huang的EMD算法起到了这样的作用，它能够将所有的时域信号转化为“线性稳态”。


# EMD (empirical mode decomposition) 经验模式分解
# IMFs (intrinsic mode function) 固有模态函数
# 对signal_data的极大值，极小值做三次样条插值，获得上包络线
# 和下包络线。然后对上下包络线做均值，然后原始信号减去该均值，
# 获得“疑似IMF”，对其进行判断：
# 条件1：均值线（总得有很多数构成吧）的平均值趋近于0（一般和0做差<0.1）
# 条件2：原始信号的极值点个数（包括极大值点个数+极小值点个数）和原始信号同y=0的交点个数之差不能大于1（小于等于1）
# 如果都满足，则该“疑似”转为真IMF。
# 
def emd(signal_data):
    # 是否单调
    def ismonotic(x):
        if (len(sig.find_peaks(x)) * len(sig.find_peaks(-x)))> 0:
            return 0
        return 1

    def has_peaks(x):
        p_max = sig.find_peaks(x)[0]
        p_min = sig.find_peaks(-x)[0]
        if len(p_max) > 3 and len(p_min) > 3:
            return True
        return False

    def getspline(x, is_max):
        # 找极大值然后三次插值，并返回
        # 会遇到三次样条插值需要4个点的情况，会报错，
        N = len(x)
        p = sig.find_peaks(x)[0]
        arr = np.arange(N)
        if p[0] != 0:
            if p[-1] != N-1:
                p = np.hstack(([0],p,[N-1]))
            else:
                p = np.hstack(([0],p))
        else:
            if p[-1] != N-1:
                p = np.hstack((p,[N-1]))
        f_new = interpolate.interp1d(arr[p], x[p], kind=3)
        return f_new(arr)

        # if is_max == True:
        #     peaks = sig.argrelmax(signal_data,)[0]
        # else:
        #     peaks = sig.argrelmin(signal_data,)[0]
        # ipo3 = interpolate.splrep(peaks, signal_data[peaks], k=3)
        # iy3 = interpolate.splev(arr, ipo3)

        # return iy3
    
    def is_imf(x):
        N = len(x)
        u1 = np.sum((x[0:N-1] * x[1:N]) < 0)
        u2 = len(sig.find_peaks(x)[0]) + len(sig.find_peaks(-x)[0])
        if abs(u1 - u2) > 1:
            return 0
        return 1

    def get_imfs(x):
        imf = []
        while(ismonotic(x) == 0 and has_peaks(x)):
            x1 = x
            sd = np.Inf
            # 条件1 或者 条件2 不成立
            while (sd > 0.1 or is_imf(x1) == 0):
                s1 = getspline(x1, True)
                s2 = -getspline(-x1, False)
                x2 = x1 - (s1 + s2)/2
                
                # 条件1
                sd = np.sum((x1-x2)**2) / np.sum(x1**2)
                x1 = x2
            # 条件1 和 条件2 成立
            imf.append(x1)
            x = x - x1
        
        imf.append(x)
        return imf

    return get_imfs(signal_data)

# 希尔伯特变换本质是一个90°的相移器
# 当频率大于0时，相位向左移90度；反之，向右移90度。
def hilbert_transform(signal_data):
    xa = sig.hilbert(signal_data)
    # 虚部就是希尔伯特信号
    hilbert_signal = np.imag(xa)
    original_signal = np.real(xa)
    return hilbert_signal


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    duration = 1.0
    fs = 400.0
    samples = int(fs*duration)
    t = np.arange(samples) / fs
    signal = (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )
    signal_new = hilbert_transform(signal)
    plt.figure()
    plt.subplot(121)
    plt.plot(signal, c="y")
    plt.subplot(122)
    plt.plot(signal_new, c="b")
    plt.show()

    # 上面是Hilbert trans 例子，下面是emd例子

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