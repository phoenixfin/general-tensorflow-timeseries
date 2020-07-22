import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from parameters import ModelPar as pm
from parameters import GeneratorPar as pg
import preprocess as pp

tf.random.set_seed(42)
np.random.seed(42)

class TimeSeriesNeuralNetwork(object):
    def __init__(self, time=None):
        self.model = tf.keras.Sequential()
        self.data = {'train': None, 'test':None}
        self.time = time if time else pg.time
        self.history = None
        self.main_series = None

    def generate_data(self, seq2seq=False):
        import generator as gen
        
        series = gen.complete(self.time, pg.baseline, pg.slope, 
                              pg.period, pg.amplitude, pg.noise_level)
        train, valid = pp.split(pg.time, series, pg.split_time)

        train_set = pp.window_dataset(train[1], pm.window_size,
                                      pm.batch_size, seq2seq=seq2seq)
        valid_set = pp.window_dataset(valid[1], pm.window_size, 
                                      pm.batch_size, seq2seq=seq2seq)        

        self.main_series = series
        self.set_data(train_set, valid_set)

    def set_data(self, train_data, test_data):
        self.data['train'] = train_data
        self.data['test'] = test_data

    def add_dim_expander(self):
        lam_func = lambda x: tf.expand_dims(x, axis=-1)
        layer = tf.keras.layers.Lambda(lam_func, input_shape=[None])
        self.model.add(layer)

    def add_recurrent_layer(self, typ, units, double=True, first=True, 
                            stateful=False, seq2seq=False):
        Layer = getattr(tf.keras.layers, typ)

        if double:
            if first:
                kwargs = {'input_shape':[None, 1]}
            if stateful:
                kwargs['batch_input_shape']=[1, None, 1]
                
            layer = Layer(units, stateful=stateful, return_sequences=True, **kwargs)
            
            self.model.add(layer)
        self.model.add(Layer(units, stateful=stateful, return_sequences=seq2seq))
    
    def add_dense(self, neurons_list, activation='relu', first=False, last=False):
        for i, neu in enumerate(neurons_list):
            kwargs = {'units': neu}
            if first and i==0:
                kwargs['input_shape'] = [pm.window_size]
                kwargs['activation'] = None
            if last and i==len(neurons_list)-1:
                kwargs['activation'] = None                
            self.model.add(tf.keras.layers.Dense(**kwargs))

    def add_convolution(self, filters, ker_size, dilation=1, first=True,
                        strides=1, padding='causal', activation='relu'):
        input_shape = [None, 1] if first else None            
        conv = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=ker_size,
            strides=strides,
            dilation_rate=dilation,
            padding=padding,
            activation=activation,
            input_shape=input_shape
        )
        self.model.add(conv)
        
    def set_input(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=[None, 1]))
            
    def close_rnn(self):
        self.add_dense([1], last=True)
        self.model.add(tf.keras.layers.Lambda(lambda x: x * 200.0))

    def show_summary(self):
        self.model.summary()
  
    def forecast(self, series):
        from preprocess import window_dataset
        ds = window_dataset(series, window_size=pm.window_size, 
                            batch_size=pm.batch_size, shuffle_buffer=0,
                            prediction=True)
        forecast = self.model.predict(ds)
        return forecast

    def train(self, cb_use):
        callbacks = self.set_callbacks(cb_use)
        optimizer = tf.keras.optimizers.SGD(lr=pm.lr_start, 
                                            momentum=pm.sgd_momentum)
        self.model.compile(loss=tf.keras.losses.Huber(),
                            optimizer=optimizer,
                            metrics=["mae"])
        history = self.model.fit(self.data['train'], 
                                epochs=pm.num_epochs, 
                                validation_data=self.data['test'], 
                                callbacks=callbacks)
        self.history = history

    def set_callbacks(self, cb_list):
        cb = tf.keras.callbacks
        
        def lr_schedule():
            func = lambda epoch: pm.lr_start * pm.lr_base**(epoch/pm.lr_divisor)
            return cb.LearningRateScheduler(func)

        def early_stopping():
            return cb.EarlyStopping(patience=pm.stop_pat)

        def model_checkpoint():
            model_checkpoint = cb.ModelCheckpoint("generated", save_best_only=True)

        def reset_states():
            class ResetStatesCallback(cb.Callback):
                def on_epoch_begin(self, epoch, logs):
                    self.model.reset_states()
            return ResetStatesCallback()        

        callbacks = []
        for cb_name in cb_list:
            func = locals()[cb_name]
            callbacks.append(func())
                
        return callbacks
    
    def plot_performance(self):
        plt.semilogx(self.history.history["lr"], self.history.history["loss"])
        plt.axis([1e-6, 1e-3, 0, 20])
        plt.show()
    
    def plot_forecast(self):
        series = self.main_series[pg.split_time - pm.window_size:-1]
        fc = self.forecast(series)[:,0]
        _, valid = pp.split(pg.time, self.main_series, pg.split_time)
        from support import plot_series
        plot_series(valid[0], [valid[1], fc], labels=["Real", "Forecast"])
