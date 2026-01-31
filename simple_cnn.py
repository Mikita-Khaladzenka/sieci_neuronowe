import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_simple_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Tworzy prosty model CNN dla klasyfikacji CIFAR-10.
    
    Args:
        input_shape: Kształt obrazów wejściowych (height, width, channels)
        num_classes: Liczba klas do klasyfikacji
    
    Returns:
        Model Keras
    """
    model = keras.Sequential([
        # Blok 1: Conv -> ReLU -> MaxPool
        layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Blok 2: Conv -> ReLU -> MaxPool
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten
        layers.Flatten(),
        
        # Fully connected layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
