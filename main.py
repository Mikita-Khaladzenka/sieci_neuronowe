"""
Główny skrypt do trenowania i testowania modelu CNN dla CIFAR-10.
Używa TensorFlow/Keras jako głównego frameworka.
"""

import tensorflow as tf
from prepare_cifar10 import prepare_data
from simple_cnn import create_simple_cnn
from train_cnn import train_model, plot_training_history
from test_cnn import test_model
from evaluate_cnn import evaluate_model, visualize_predictions

# Nazwy klas CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def main():
    print("="*60)
    print("PROJEKT: Klasyfikacja obrazów CIFAR-10 z TensorFlow/Keras")
    print("="*60)
    
    # Sprawdzenie dostępności GPU
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPU dostępne: {len(tf.config.list_physical_devices('GPU')) > 0}")
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f"Urządzenia GPU: {tf.config.list_physical_devices('GPU')}")
    
    # Przygotowanie danych
    print("\n" + "-"*60)
    print("PRZYGOTOWYWANIE DANYCH")
    print("-"*60)
    train_dataset, val_dataset, test_dataset, _, _, _ = prepare_data(
        val_split=0.1, 
        seed=42, 
        batch_size=32
    )
    
    # Tworzenie modelu
    print("\n" + "-"*60)
    print("TWORZENIE MODELU")
    print("-"*60)
    model = create_simple_cnn(input_shape=(32, 32, 3), num_classes=10)
    model.summary()
    
    # Trening modelu
    print("\n" + "-"*60)
    print("TRENING MODELU")
    print("-"*60)
    trained_model, history = train_model(
        model, 
        train_dataset, 
        val_dataset, 
        epochs=10, 
        learning_rate=0.001
    )
    
    # Wizualizacja historii treningu
    print("\n" + "-"*60)
    print("WIZUALIZACJA HISTORII TRENINGU")
    print("-"*60)
    plot_training_history(history)
    
    # Testowanie modelu
    print("\n" + "-"*60)
    print("TESTOWANIE MODELU")
    print("-"*60)
    test_results = test_model(trained_model, test_dataset)
    
    # Szczegółowa ewaluacja
    print("\n" + "-"*60)
    print("SZCZEGÓŁOWA EWALUACJA")
    print("-"*60)
    evaluate_model(trained_model, test_dataset, class_names=class_names)
    
    # Wizualizacja przykładowych predykcji
    print("\n" + "-"*60)
    print("WIZUALIZACJA PRZYKŁADOWYCH PREDYKCJI")
    print("-"*60)
    visualize_predictions(trained_model, test_dataset, class_names=class_names, num_samples=16)
    
    print("\n" + "="*60)
    print("PROCES ZAKOŃCZONY POMYŚLNIE!")
    print("="*60)

if __name__ == '__main__':
    main()
