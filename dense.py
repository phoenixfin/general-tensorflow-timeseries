from basic import TimeSeriesNeuralNetwork

class TimeSeriesDense(TimeSeriesNeuralNetwork):
    def __init__(self):
        super().__init__()
        self.generate_data()
        
    def main_linear(self):
        self.add_dense([1], first=True)
        self.show_summary()
        self.train(cb_use = ['lr_schedule', 'early_stopping'])
        self.plot_performance()
        self.plot_forecast()
        
    def main_deep(self):
        self.add_dense([10,10,1], first=True, last=True)
        self.show_summary()        
        self.train(cb_use = ['lr_schedule', 'early_stopping'])
        self.plot_forecast()