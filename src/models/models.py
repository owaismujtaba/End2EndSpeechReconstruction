from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GRU, Dense, Flatten,
    Reshape, concatenate, Input
)
import config as config

class NeuroInceptDecoder(tf.keras.Model):
    def __init__(self, n_classes, n_channels, n_features):
        super(NeuroInceptDecoder, self).__init__()
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_channels = n_channels

        self.conv1x1 = Conv1D(64, 1, padding='same', activation='relu')
        self.conv3x3 = Conv1D(64, 3, padding='same', activation='relu')
        self.conv5x5 = Conv1D(64, 5, padding='same', activation='relu')
        self.maxpool = MaxPooling1D(3, strides=1, padding='same')
        self.maxpool_conv = Conv1D(64, 1, padding='same', activation='relu')

        self.gru1 = GRU(128, return_sequences=True)
        self.gru2 = GRU(256, return_sequences=True)
        self.gru3 = GRU(512, return_sequences=False)

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


    def train_step(self, X, y):
        with tf.GradientTape() as tape:
            y_pred = self(X)
            loss = self.compiled_loss(y, y_pred)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def eval(self, X_val, y_val):
        y_pred = self(X_val)
        loss = self.compiled_loss(y_val, y_pred)
        return loss

    def train(self, X, y, batch_size=32, epochs=config.EPOCHS, learning_rate=0.001, val_size=0.2):
        self.optimizer = Adam(learning_rate)
        self.compiled_loss = tf.keras.losses.MeanSquaredError()

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = X_train.shape[0] // batch_size

            indices = tf.random.shuffle(tf.range(X_train.shape[0]))
            X_train_shuffled = tf.gather(X_train, indices)
            y_train_shuffled = tf.gather(y_train, indices)

            for i in range(num_batches):
                batch_X = X_train_shuffled[i * batch_size : (i + 1) * batch_size]
                batch_y = y_train_shuffled[i * batch_size : (i + 1) * batch_size]

                loss = self.train_step(batch_X, batch_y)
                epoch_loss += loss

            epoch_loss /= num_batches

            val_loss = self.eval(X_val, y_val)
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss.numpy()}, Validation Loss: {val_loss.numpy()}')