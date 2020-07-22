
import numpy as np
import matplotlib.pyplot as plt

import generator as gen
import preprocess as pp
from parameters import GeneratorPar as p
from support import plot_series, mae

def naive(series, split_time):
    return series[split_time - 1:-1]


def moving_average(series, window_size):
    mov = np.cumsum(series)
    mov[window_size:] = mov[window_size:] - mov[:-window_size]
    return mov[window_size - 1:-1] / window_size

series = gen.complete(p.time, p.baseline, p.slope, 
                      p.period, p.amplitude, p.noise_level)
time2, series2 = pp.remove_season(time, series)

train, valid = pp.split(time, series, split_time)
train2, valid2 = pp.split(time, series2, split_time-period)

naive_prediction = naive(series, split_time)
plot_series(valid[0], [valid[1], naive_prediction], labels=["Series", "Naive Forecast"])
print(mae(naive_prediction, valid[1]))

window = 30
moving_avg = moving_average(series, window)[split_time - window:]
plot_series(valid[0], [valid[1], moving_avg], labels=["Series", "Moving average (30 days)"])
print(mae(moving_avg, valid[1]))


window = 50
diff_moving_avg = moving_average(series2, window)[split_time - period - window:]

plot_series(valid[0], [valid2[1], diff_moving_avg], 
            labels=["Series(t) – Series(t–365)", "Moving Average of Diff"])
print(mae(diff_moving_avg, valid2[1]))


diff_moving_avg2 = series[split_time - 365:-365] + diff_moving_avg
plot_series(valid[0], [valid[1], diff_moving_avg2], 
            labels=["Series(t) – Series(t–365)", "Moving Average of Diff plus past"])
print(mae(diff_moving_avg2, valid[1]))


diff_moving_avg3 = moving_average(series[split_time - 370:-359], 11) + diff_moving_avg
plot_series(valid[0], [valid[1], diff_moving_avg3], 
            labels=["Series(t) – Series(t–365)", "Moving Average of Diff plus smooth past"])
print(mae(diff_moving_avg3, valid[1]))
