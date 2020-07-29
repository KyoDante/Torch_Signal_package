import librosa
from librosa.feature import mfcc

"""
y : np.ndarray [shape=(n,)] or None audio time series

sr : number > 0 [scalar] sampling rate of y

S : np.ndarray [shape=(d, t)] or None log-power Mel spectrogram

n_mfcc: int > 0 [scalar] number of MFCCs to return

dct_type : {1, 2, 3} Discrete cosine transform (DCT) type. By default, DCT type-2 is used.

norm : None or 'ortho' If dct_type is 2 or 3, setting norm='ortho' uses an ortho-normal DCT basis.

Normalization is not supported for `dct_type=1`.
lifter : number >= 0 If lifter>0, apply *liftering* (cepstral filtering) to the MFCCs:

`M[n, :] <- M[n, :] * (1 + sin(pi * (n + 1) / lifter)) * lifter / 2`

Setting `lifter >= 2 * n_mfcc` emphasizes the higher-order coefficients.
As `lifter` increases, the coefficient weighting becomes approximately linear.
"""
# **kargs 需要控制更多内容，比如计算梅尔频谱时候的nfft和步长。
def mfcc_(signal_data, sampling_rate, **kargs):
    mfcc_features = mfcc(signal_data, sampling_rate)
    return mfcc_features
    

if __name__ == "__main__":
    y, sr = librosa.load(librosa.util.example_audio_file(), offset=30, duration=5)
    mfcc_fea = mfcc_(y, sr)
    import numpy as np
    print(len(y), sr)
    print(np.shape(mfcc_fea))
    print(mfcc_fea)