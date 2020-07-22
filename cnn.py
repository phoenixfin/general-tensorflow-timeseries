from basic import TimeSeriesNeuralNetwork

class TimeSeriesConvolutional(TimeSeriesNeuralNetwork):
    def __init__(self):
        super().__init__()
        self.generate_data()
    
    def main(self):
        self.add_convolution(32, 5, first=True)
        self.add_recurrent_layer('LSTM', 32, seq2seq=True)
        self.wrap_up()

    def main_fully(self):
        self.add_convolution(32, 2, first=True)
        for dilation_rate in (2, 4, 8, 16, 32):
            self.add_convolution(32, 2, dilation=dilation_rate)
        self.add_convolution(1,1)
        self.wrap_up()

    def wrap_up(self):
        self.close_rnn()
        self.show_summary()
        self.train(cb_use = ['lr_schedule', 'early_stopping'])
        self.plot_forecast()