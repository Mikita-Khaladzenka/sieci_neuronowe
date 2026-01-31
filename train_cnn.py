import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_dataset, val_dataset, epochs=10, learning_rate=0.001, verbose=1):
    """
    Trenuje model CNN używając TensorFlow/Keras.
    
    Args:
        model: Model Keras do wytrenowania
        train_dataset: TensorFlow dataset z danymi treningowymi
        val_dataset: TensorFlow dataset z danymi walidacyjnymi
        epochs: Liczba epok treningu
        learning_rate: Współczynnik uczenia
        verbose: Poziom szczegółowości (0, 1, 2)
    
    Returns:
        Wytrenowany model oraz historia treningu
    """
    # Kompilacja modelu
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callback do zapisywania najlepszego modelu
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Callback do redukcji learning rate
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # Trening modelu
    print("Rozpoczynam trening modelu...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback, reduce_lr],
        verbose=verbose
    )
    
    print('Trening zakończony!')
    return model, history


def plot_training_history(history, save_path='training_history.png'):
    """
    Wizualizuje historię treningu używając matplotlib.
    
    Args:
        history: Obiekt History zwrócony przez model.fit()
        save_path: Ścieżka do zapisania wykresu
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Wykres dokładności
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Wykres straty
    axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Wykres zapisany w: {save_path}")
    plt.show()
