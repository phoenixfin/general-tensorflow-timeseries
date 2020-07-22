import numpy as np
import matplotlib.pyplot as plt

def plot_series(time, series_list, format="-", start=0, end=None, labels=None):
    plt.figure(figsize=(10, 6))

    plt.xlabel("Time")
    plt.ylabel("Value")
    for series, label in zip(series_list, labels):
        plt.plot(time[start:end], series[start:end], format, label=label)
    if labels:
        plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()    

def mae(predicted, true_val):
    errors = predicted - true_val
    abs_errors = np.abs(errors)
    mae = abs_errors.mean()
    return mae
