import numpy as np

# 数据格式为
# Time series:
# x1, x2, x3, x4,...

def confirm_np_array(time_series):
    if not isinstance(time_series, np.ndarray):
        time_series = np.array(time_series)
    return time_series

# mean
def mean(time_series):
    time_series = confirm_np_array(time_series)
    return np.mean(time_series)

# variance
def var(time_series):
    time_series = confirm_np_array(time_series)
    return np.var(time_series)

# standard variance
def std(time_series):
    time_series = confirm_np_array(time_series)
    return np.std(time_series)

# 偏度（skewness）也称为偏态、偏态系数，是统计数据分布偏斜方向和程度的度量，
# 是统计数据分布非对称程度的数字特征。
# ref: https://baike.baidu.com/item/%E5%81%8F%E5%BA%A6/8626571?fr=aladdin
def skewness(time_series):
    return np.mean(np.power((time_series - np.mean(time_series))/np.std(time_series), 3))

# 峰度（peakedness;kurtosis）又称峰态系数。表征概率密度分布曲线在平均值处峰值高低的特征数。
# 直观看来，峰度反映了峰部的尖度。样本的峰度是和正态分布相比较而言统计量，
# 如果峰度大于三，峰的形状比较尖，比正态分布峰要陡峭。反之亦然。
# ref: https://baike.baidu.com/item/峰度
def kurtosis(time_series):
    return np.sum(np.power(time_series - np.mean(time_series), 4)) / np.power(np.std(time_series), 4)

def zero_crossing_rate(time_series):
    def sgn(time_series):
        for idx, num in enumerate(time_series):
            if num >=0:
                time_series[idx] = 1
            else:
                time_series[idx] = -1
        return time_series
    time_series = confirm_np_array(time_series)
    time_series = sgn(time_series)
    zcr = 0.5 * np.sum(np.abs(time_series[1:]-time_series[0:-1]))
    return zcr


if __name__ == "__main__":
    print(zero_crossing_rate([0,1,2,3,2,1,0,-1,2,]))
    print(skewness([0,1,2,3,2,1,0,-1,2,]))