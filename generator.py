import numpy as np

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def complete(time, baseline=0, slope=0, period=0, amplitude=0, noise_level=0):
    trn = trend(time, slope)
    season = seasonality(time, period, amplitude)
    noise = white_noise(time, noise_level, seed=42)
    
    return baseline + trn + season + noise
    