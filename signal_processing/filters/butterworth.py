import scipy.signal as sig
import scipy.fft as fft

# butterworth filter
def butterworth(sampling_rate, order, filter_type, freq_range):
    filtered = None
    if filter_type == 'band_pass':
        sos = sig.butter(order, freq_range, 'bandpass', fs=sampling_rate, output='sos')
        filtered = sig.sosfilt(sos, sig)
    elif filter_type == 'band_stop':
        sos = sig.butter(order, freq_range, 'bandstop', fs=sampling_rate, output='sos')
        filtered = sig.sosfilt(sos, sig)
    elif filter_type == 'high_pass':
        sos = sig.butter(order, freq_range, 'highpass', fs=sampling_rate, output='sos')
        filtered = sig.sosfilt(sos, sig)
    elif filter_type == 'low_pass':
        sos = sig.butter(order, freq_range, 'lowpass', fs=sampling_rate, output='sos')
        filtered = sig.sosfilt(sos, sig)
    return filtered

if __name__ == "__main__":
    butterworth(44100, 6, 'bandpass', [17700, 18300])