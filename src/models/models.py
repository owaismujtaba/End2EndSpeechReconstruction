from pathlib import Path
import tensorflow as tf

class NeuroInceptDecoder(tf.keras.Model):
    def __init__(self, n_classes, n_channels, n_timepoints):
        super(NeuroInceptDecoder, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

    
    def inception_module(self, input, filters):
        pass