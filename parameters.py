import numpy as np

class GeneratorPar():
    time = np.arange(4 * 365 + 1)
    baseline = 10
    amplitude = 40
    period = 365
    slope = 0.05
    noise_level = 5
    split_time = 1000
    period = 365

class ModelPar():
    window_size = 30
    num_epochs = 100
    lr_start = 1e-7
    lr_base = 10
    lr_divisor = 20
    stop_pat = 50
    sgd_momentum = .9
    batch_size = 1
    