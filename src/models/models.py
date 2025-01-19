from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GRU, Dense, Flatten,
    Reshape, concatenate, Input
)
import config as config
import pandas as pd
from tqdm import tqdm


early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  
        patience=5,         
        restore_best_weights=True  
    )

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

    def train_step(self, data):
        X, y = data

        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred)  # Compute loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def eval(self, X_val, y_val):
        y_pred = self(X_val)
        loss = self.compiled_loss(y_val, y_pred)
        return loss

    def train(self, X, y, learning_rate=0.001, val_size=0.2):
        print("🔧 Starting Model Training Process 🔧")
        print(f"🟢 Input Data Shapes: X={X.shape}, y={y.shape}")
        print(f"🔄 Splitting data with validation size: {val_size}")

        print("⚙️ Compiling the model...")
        self.compile(
            optimizer=Adam(learning_rate),
            loss=tf.keras.losses.MeanSquaredError(name='loss'),
            metrics=['accuracy', tf.keras.metrics.MeanAbsoluteError(name='mae')]
        )
        print(f"✅ Model compiled with learning_rate={learning_rate}")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=val_size,
            shuffle=True,
            random_state=42  
        )
        print(f"📊 Training Data Shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"📊 Validation Data Shapes: X_val={X_val.shape}, y_val={y_val.shape}")

        print("🚀 Training the model...")
        history = self.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            verbose=1,
            callbacks=[
                early_stopping, 
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='best_model.h5',
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    verbose=1,
                    min_lr=1e-6
                )
            ]
        )
        print("✅ Model training completed")

        history_df = pd.DataFrame(history.history)
        print(f"📈 Training history keys: {list(history.history.keys())}")
        print(f"💾 First few rows of training history:\n{history_df.head()}")

        return history_df

        