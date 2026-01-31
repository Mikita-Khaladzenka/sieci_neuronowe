import tensorflow as tf
from tensorflow import keras
import numpy as np

def test_model(model, test_dataset, verbose=1):
    """
    Testuje model na zbiorze testowym.
    
    Args:
        model: Wytrenowany model Keras
        test_dataset: TensorFlow dataset z danymi testowymi
        verbose: Poziom szczegółowości
    
    Returns:
        Słownik z wynikami testów (loss, accuracy)
    """
    print("Testowanie modelu na zbiorze testowym...")
    
    results = model.evaluate(test_dataset, verbose=verbose)
    
    test_loss = results[0]
    test_accuracy = results[1]
    
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy*100:.2f}%')
    
    return {
        'loss': test_loss,
        'accuracy': test_accuracy
    }
