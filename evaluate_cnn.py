import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_model(model, test_dataset, class_names=None, save_path='evaluation_results.png'):
    """
    Rozszerzona funkcja ewaluacji modelu z metrykami i wizualizacjami.
    
    Args:
        model: Wytrenowany model Keras
        test_dataset: TensorFlow dataset z danymi testowymi
        class_names: Lista nazw klas (opcjonalnie)
        save_path: Ścieżka do zapisania wykresów
    """
    print("Ewaluacja modelu...")
    
    # Zbieranie predykcji i prawdziwych etykiet
    y_true = []
    y_pred = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Macierz pomyłek
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print("MACIERZ POMYŁEK")
    print("="*50)
    print(cm)
    
    # Raport klasyfikacji
    print("\n" + "="*50)
    print("RAPORT KLASYFIKACJI")
    print("="*50)
    if class_names:
        print(classification_report(y_true, y_pred, target_names=class_names))
    else:
        print(classification_report(y_true, y_pred))
    
    # Wizualizacja macierzy pomyłek
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)))
    plt.title('Macierz pomyłek (Confusion Matrix)')
    plt.ylabel('Prawdziwe etykiety')
    plt.xlabel('Przewidywane etykiety')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nWykres macierzy pomyłek zapisany w: {save_path}")
    plt.show()
    
    return {
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }


def visualize_predictions(model, test_dataset, class_names=None, num_samples=16, save_path='sample_predictions.png'):
    """
    Wizualizuje przykładowe predykcje modelu.
    
    Args:
        model: Wytrenowany model Keras
        test_dataset: TensorFlow dataset z danymi testowymi
        class_names: Lista nazw klas (opcjonalnie)
        num_samples: Liczba przykładów do wyświetlenia
        save_path: Ścieżka do zapisania wykresu
    """
    # Pobranie próbki danych
    images_batch = []
    labels_batch = []
    
    for images, labels in test_dataset.take(1):
        images_batch = images.numpy()[:num_samples]
        labels_batch = labels.numpy()[:num_samples]
        break
    
    # Predykcje
    predictions = model.predict(images_batch, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence = np.max(predictions, axis=1)
    
    # Wizualizacja
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        axes[i].imshow(images_batch[i])
        
        true_label = labels_batch[i]
        pred_label = predicted_classes[i]
        conf = confidence[i]
        
        true_name = class_names[true_label] if class_names else f'Class {true_label}'
        pred_name = class_names[pred_label] if class_names else f'Class {pred_label}'
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_name}\nPred: {pred_name} ({conf:.2f})', 
                         color=color, fontsize=8)
        axes[i].axis('off')
    
    # Ukryj puste subploty
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Wizualizacja predykcji zapisana w: {save_path}")
    plt.show()
