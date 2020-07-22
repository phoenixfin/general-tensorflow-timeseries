import tensorflow as tf

def split(time, series, split_time):
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    
    return (time_train, x_train), (time_valid, x_valid)
    
def remove_season(time, series, period=365):
    diff_series = (series[period:] - series[:-period])
    diff_time = time[period:]
    return diff_time, diff_series

def window_dataset(series, window_size, batch_size=32,
                   shuffle_buffer=1000, seq2seq=False, prediction=False):
    win_size = window_size if prediction else window_size + 1
    if seq2seq:
        series = tf.expand_dims(series, axis=-1)
        shift = window_size
    else:
        shift = 1
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(win_size, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(win_size))
    if shuffle_buffer and not seq2seq:
        dataset = dataset.shuffle(shuffle_buffer)        
    if not prediction:
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

