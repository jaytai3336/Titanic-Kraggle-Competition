import tensorflow as tf
from keras.models import Sequential, load_model as keras_load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import datetime as dt

class Model:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=input_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification

        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.config['training']['learning_rate']),
            metrics=self.config['model']['metrics']
        )

        self.model = model
        print("[Model] Feedforward NN built and compiled.")
        return model

    def train(self, x, y, validation_data=None):
        save_dir = self.config['model']['save_dir']
        os.makedirs(os.path.join(save_dir, 'archive'), exist_ok=True)

        save_path = os.path.join(
            save_dir, 'archive',
            f"{dt.datetime.now():%d%m%Y-%H%M%S}-e{self.config['training']['epochs']}.keras"
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5),
            ModelCheckpoint(save_path, save_best_only=True, verbose=1)
        ]

        self.history = self.model.fit(
            x, y,
            validation_data=validation_data,
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            callbacks=callbacks
        )
        self.model.load_weights(save_path)
        print(f"[Model] Training completed and best weights loaded from {save_path}.")

    def evaluate(self, x_test, y_test):
        print("[Model] Evaluating model...")
        results = self.model.evaluate(x_test, y_test)
        for name, value in zip(self.model.metrics_names, results):
            print(f"{name}: {value:.4f}")

    def predict(self, x, verbose=1):
        return self.model.predict(x, verbose=verbose)

    def summary(self):
        self.model.summary()

    def save_model(self, path=None):
        """Save the full model to the specified path or default directory."""
        if path is None:
            save_dir = self.config['model']['save_dir']
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, 'model.keras')

        self.model.save(path)
        print(f"[Model] Full model saved to {path}.")

    def load_model(self, path=None):
        """Load a saved Keras model from disk."""
        if path is None:
            path = os.path.join(self.config['model']['save_dir'], 'model.keras')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        
        self.model = keras_load_model(path)
        print(f"[Model] Model loaded from {path}.")
