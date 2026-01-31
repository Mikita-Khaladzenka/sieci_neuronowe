import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, EfficientNetB0

def create_resnet50_model(input_shape=(32, 32, 3), num_classes=10, pretrained=True):
    """
    Tworzy model oparty na ResNet50 z transfer learning.
    
    Args:
        input_shape: Kształt obrazów wejściowych
        num_classes: Liczba klas do klasyfikacji
        pretrained: Czy użyć wstępnie wytrenowanych wag ImageNet
    
    Returns:
        Model Keras
    """
    # ResNet50 wymaga obrazów o rozmiarze co najmniej 32x32
    # Dla CIFAR-10 (32x32) możemy użyć ResNet50, ale lepiej przeskalować do 224x224
    # lub użyć EfficientNetB0 który lepiej radzi sobie z mniejszymi obrazami
    
    base_model = ResNet50(
        weights='imagenet' if pretrained else None,
        include_top=False,
        input_shape=(224, 224, 3)  # ResNet50 wymaga większych obrazów
    )
    
    # Zamrożenie warstw bazowych (opcjonalnie)
    if pretrained:
        base_model.trainable = False
    
    model = keras.Sequential([
        layers.Lambda(lambda x: tf.image.resize(x, (224, 224))),  # Resize dla ResNet50
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_efficientnet_model(input_shape=(32, 32, 3), num_classes=10, pretrained=True):
    """
    Tworzy model oparty na EfficientNetB0 z transfer learning.
    EfficientNet lepiej radzi sobie z mniejszymi obrazami niż ResNet.
    
    Args:
        input_shape: Kształt obrazów wejściowych
        num_classes: Liczba klas do klasyfikacji
        pretrained: Czy użyć wstępnie wytrenowanych wag ImageNet
    
    Returns:
        Model Keras
    """
    base_model = EfficientNetB0(
        weights='imagenet' if pretrained else None,
        include_top=False,
        input_shape=(224, 224, 3)  # EfficientNet również preferuje większe obrazy
    )
    
    # Zamrożenie warstw bazowych (opcjonalnie)
    if pretrained:
        base_model.trainable = False
    
    model = keras.Sequential([
        layers.Lambda(lambda x: tf.image.resize(x, (224, 224))),  # Resize dla EfficientNet
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_efficientnet_small_model(input_shape=(32, 32, 3), num_classes=10, pretrained=True):
    """
    Tworzy model EfficientNet dostosowany do mniejszych obrazów (32x32).
    Używa mniejszego modelu bazowego lub dostosowuje architekturę.
    
    Args:
        input_shape: Kształt obrazów wejściowych
        num_classes: Liczba klas do klasyfikacji
        pretrained: Czy użyć wstępnie wytrenowanych wag ImageNet
    
    Returns:
        Model Keras
    """
    # Dla obrazów 32x32 możemy użyć EfficientNet z resize lub stworzyć własną architekturę
    base_model = EfficientNetB0(
        weights='imagenet' if pretrained else None,
        include_top=False,
        input_shape=(32, 32, 3)  # Spróbujmy bezpośrednio z 32x32
    )
    
    if pretrained:
        base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
