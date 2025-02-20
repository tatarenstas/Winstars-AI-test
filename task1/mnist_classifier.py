from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np

#interface for classification models
class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass

#random forest
class RandomForestModel(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
    
    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

#feed-forward neural  network
class FeedForwardNeuralNetworkModel(MnistClassifierInterface):
    def __init__(self):
        self.model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(784,)),
            layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=5, batch_size=32)

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)

#convolutional neural network
class CNNModel(MnistClassifierInterface):
    def __init__(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        X_train_scaled = X_train / 255.0
        self.model.fit(X_train_scaled, y_train, epochs=5)
    
    def predict(self, X_test):
        X_test_scaled = X_test / 255.0
        return self.model.predict(X_test_scaled).argmax(axis=1)

#main classifier class to select  the model
class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == 'rf':
            self.model = RandomForestModel()
        elif algorithm == 'ffnn':
            self.model = FeedForwardNeuralNetworkModel()
        elif algorithm == 'cnn':
            self.model = CNNModel()
        else:
            raise ValueError("Algorithm must be 'rf', 'ffnn', or 'cnn'")

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)