import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_cnn_model(input_shape=(256, 256, 1), n_classes=4):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        
        layers.Dense(24, activation='relu'),  # Changed from 32 to 24
        layers.Dropout(0.5),
        layers.Dense(24, activation='relu'),  # Changed from 32 to 24
        layers.Dropout(0.5),
        
        layers.Dense(n_classes, activation='softmax')
    ])
    return model
