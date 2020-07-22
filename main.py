from dense import TimeSeriesDense
from rnn import TimeSeriesRecurrent

if __name__ == '__main__':
    # ts = TimeSeriesDense()
    # ts.main_deep()
    
    ts = TimeSeriesRecurrent()
    ts.main_lstm()
    

# rnn_forecast = model_forecast(model, series[:,  np.newaxis], window_size)
# rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

# cnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
# cnn_forecast = cnn_forecast[split_time - window_size:-1, -1, 0]

# rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
# rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]


# rnn_forecast = model.predict(series[np.newaxis, :, np.newaxis])
# rnn_forecast = rnn_forecast[0, split_time - 1:-1, 0]

# rnn_forecast.shape




# keras.metrics.mean_absolute_error(x_valid, lin_forecast).numpy()
