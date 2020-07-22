from basic import TimeSeriesNeuralNetwork

class TimeSeriesRecurrent(TimeSeriesNeuralNetwork):
    def __init__(self):
        super().__init__()
        self.cb = ['lr_schedule', 'early_stopping']
                
    def main(self):
        self.generate_data()        
        self.add_dim_expander()
        self.add_recurrent_layer('SimpleRNN', units=100)
        self.wrap_up()
    
    def main_seq2seq(self):
        self.generate_data()        
        self.add_recurrent_layer('SimpleRNN', units=100, 
                                 first=True, seq2seq=True)
        self.wrap_up()
  
    def main_stateful(self):
        self.generate_data(seq2seq=True)
        self.add_recurrent_layer('SimpleRNN', units=100, first=True,
                                stateful=True, seq2seq=True)
        self.cb.append('reset_states')
        self.wrap_up()
        
    def main_lstm(self):
        self.generate_data(seq2seq=True)
        self.add_recurrent_layer('LSTM', units=100, first=True,
                                stateful=True, seq2seq=True)
        self.cb.append('reset_states')
        self.wrap_up()
        
    def wrap_up(self):
        self.close_rnn()
        self.show_summary()
        self.train(cb_use = self.cb)
        self.plot_forecast()
