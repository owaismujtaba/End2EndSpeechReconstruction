from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GRU, Dense, Flatten,
    Reshape, concatenate, Input
)

class NeuroInceptDecoder(tf.keras.Model):
    def __init__(self, n_classes, n_channels, n_features):
        super(NeuroInceptDecoder, self).__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_channels = n_channels

        # Define layers for the inception module
        self.conv1x1 = Conv1D(64, 1, padding='same', activation='relu')
        self.conv3x3 = Conv1D(64, 3, padding='same', activation='relu')
        self.conv5x5 = Conv1D(64, 5, padding='same', activation='relu')
        self.maxpool = MaxPooling1D(3, strides=1, padding='same')
        self.maxpool_conv = Conv1D(64, 1, padding='same', activation='relu')

        # Define GRU layers
        self.gru1 = GRU(128, return_sequences=True)
        self.gru2 = GRU(256, return_sequences=True)
        self.gru3 = GRU(512, return_sequences=False)

        # Define additional layers
        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(1024, activation='relu')
        self.dense3 = Dense(512, activation='relu')
        self.dense4 = Dense(256, activation='relu')
        self.dense5 = Dense(128, activation='relu')
        self.output_layer = Dense(self.n_classes, activation='linear')

    def inception_module(self, input_tensor):
        conv_1x1 = self.conv1x1(input_tensor)
        conv_3x3 = self.conv3x3(input_tensor)
        conv_5x5 = self.conv5x5(input_tensor)

        max_pool = self.maxpool(input_tensor)
        max_pool = self.maxpool_conv(max_pool)

        output = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=-1)
        return output

    def call(self, inputs):
        x = self.inception_module(inputs)

        x = self.gru1(x)
        x = self.gru2(x)
        x = self.gru3(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

        output = self.output_layer(x)

        return output
