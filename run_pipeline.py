"""
Skrypt do uruchamiania pełnego pipeline'u treningu z różnymi modelami.
Obsługuje zarówno modele trenowane od zera, jak i transfer learning.
"""

import tensorflow as tf
from prepare_cifar10 import prepare_data
from simple_cnn import create_simple_cnn
from transfer_learning_models import create_efficientnet_model, create_resnet50_model
from train_cnn import train_model, plot_training_history
from test_cnn import test_model
from evaluate_cnn import evaluate_model

# Nazwy klas CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def run_simple_cnn():
    """Uruchamia pipeline z prostym modelem CNN."""
    print("\n" + "="*60)
    print("MODEL: Prosty CNN (od zera)")
    print("="*60)
    
    train_dataset, val_dataset, test_dataset, _, _, _ = prepare_data(
        val_split=0.1, 
        seed=42, 
        batch_size=32
    )
    
    model = create_simple_cnn(input_shape=(32, 32, 3), num_classes=10)
    model.summary()
    
    trained_model, history = train_model(
        model, 
        train_dataset, 
        val_dataset, 
        epochs=10, 
        learning_rate=0.001
    )
    
    plot_training_history(history, save_path='simple_cnn_history.png')
    test_results = test_model(trained_model, test_dataset)
    evaluate_model(trained_model, test_dataset, class_names=class_names, 
                   save_path='simple_cnn_confusion_matrix.png')
    
    return trained_model, history, test_results


def run_transfer_learning_efficientnet():
    """Uruchamia pipeline z EfficientNet (transfer learning)."""
    print("\n" + "="*60)
    print("MODEL: EfficientNetB0 (Transfer Learning)")
    print("="*60)
    
    train_dataset, val_dataset, test_dataset, _, _, _ = prepare_data(
        val_split=0.1, 
        seed=42, 
        batch_size=16  # Mniejszy batch dla większych modeli
    )
    
    model = create_efficientnet_model(
        input_shape=(32, 32, 3), 
        num_classes=10, 
        pretrained=True
    )
    model.summary()
    
    trained_model, history = train_model(
        model, 
        train_dataset, 
        val_dataset, 
        epochs=5,  # Mniej epok dla transfer learning
        learning_rate=0.0001  # Mniejszy learning rate
    )
    
    plot_training_history(history, save_path='efficientnet_history.png')
    test_results = test_model(trained_model, test_dataset)
    evaluate_model(trained_model, test_dataset, class_names=class_names,
                   save_path='efficientnet_confusion_matrix.png')
    
    return trained_model, history, test_results


def run_transfer_learning_resnet():
    """Uruchamia pipeline z ResNet50 (transfer learning)."""
    print("\n" + "="*60)
    print("MODEL: ResNet50 (Transfer Learning)")
    print("="*60)
    
    train_dataset, val_dataset, test_dataset, _, _, _ = prepare_data(
        val_split=0.1, 
        seed=42, 
        batch_size=16  # Mniejszy batch dla większych modeli
    )
    
    model = create_resnet50_model(
        input_shape=(32, 32, 3), 
        num_classes=10, 
        pretrained=True
    )
    model.summary()
    
    trained_model, history = train_model(
        model, 
        train_dataset, 
        val_dataset, 
        epochs=5,  # Mniej epok dla transfer learning
        learning_rate=0.0001  # Mniejszy learning rate
    )
    
    plot_training_history(history, save_path='resnet_history.png')
    test_results = test_model(trained_model, test_dataset)
    evaluate_model(trained_model, test_dataset, class_names=class_names,
                   save_path='resnet_confusion_matrix.png')
    
    return trained_model, history, test_results


def main():
    """Główna funkcja uruchamiająca wybrane modele."""
    print("="*60)
    print("PIPELINE TRENINGU MODELI CNN DLA CIFAR-10")
    print("Framework: TensorFlow/Keras")
    print("="*60)
    
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPU dostępne: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Wybór modelu do uruchomienia
    print("\nDostępne modele:")
    print("1. Prosty CNN (od zera)")
    print("2. EfficientNetB0 (Transfer Learning)")
    print("3. ResNet50 (Transfer Learning)")
    print("4. Wszystkie modele")
    
    choice = input("\nWybierz model (1-4): ").strip()
    
    if choice == '1':
        run_simple_cnn()
    elif choice == '2':
        run_transfer_learning_efficientnet()
    elif choice == '3':
        run_transfer_learning_resnet()
    elif choice == '4':
        print("\nUruchamianie wszystkich modeli...")
        run_simple_cnn()
        run_transfer_learning_efficientnet()
        run_transfer_learning_resnet()
    else:
        print("Nieprawidłowy wybór. Uruchamiam domyślny model (Prosty CNN).")
        run_simple_cnn()
    
    print("\n" + "="*60)
    print("PIPELINE ZAKOŃCZONY!")
    print("="*60)


if __name__ == '__main__':
    main()
