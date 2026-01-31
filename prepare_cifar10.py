import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

# === 1. Ścieżka do folderu z plikami CIFAR-10 (Python version) ===
DATA_DIR = "cifar-10-python/cifar-10-batches-py"


# === 2. Wczytanie pojedynczego batcha ===
def load_batch(file_path: str):
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"]          # shape: (10000, 3072)
    labels = batch[b"labels"]        # list length 10000
    return images, labels


# === 3. Wczytanie całego treningu ===
def load_training_data():
    all_images = []
    all_labels = []
    for i in range(1, 6):
        file = os.path.join(DATA_DIR, f"data_batch_{i}")
        images, labels = load_batch(file)
        all_images.append(images)
        all_labels.extend(labels)
    x = np.concatenate(all_images, axis=0)
    y = np.array(all_labels, dtype=np.int64)
    return x, y


# === 4. Wczytanie testu ===
def load_test_data():
    x, y = load_batch(os.path.join(DATA_DIR, "test_batch"))
    return x, np.array(y, dtype=np.int64)


# === 5. Konwersja do obrazu RGB (N, 32, 32, 3) ===
def convert_images(raw_images: np.ndarray) -> np.ndarray:
    raw_images = raw_images.reshape(-1, 3, 32, 32)      # (N, 3, 32, 32)
    images = raw_images.transpose(0, 2, 3, 1)           # (N, 32, 32, 3)
    return images


def prepare_data(val_split: float = 0.1, seed: int = 42, batch_size: int = 32):
    """
    Przygotowuje dane CIFAR-10 i zwraca TensorFlow datasets.
    
    Zwraca:
      train_dataset, val_dataset, test_dataset (tf.data.Dataset)
    oraz
      (x_train, y_train), (x_val, y_val), (x_test, y_test) jako numpy arrays
    """
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Nie znaleziono folderu z CIFAR-10: {DATA_DIR}. "
            f"Upewnij się, że masz pliki cifar-10-batches-py w projekcie."
        )

    print("Wczytywanie danych treningowych (NumPy/pickle)...")
    x_train_raw, y_train = load_training_data()
    x_train = convert_images(x_train_raw).astype(np.float32) / 255.0

    print("Wczytywanie danych testowych (NumPy/pickle)...")
    x_test_raw, y_test = load_test_data()
    x_test = convert_images(x_test_raw).astype(np.float32) / 255.0

    # Podział train/val
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x_train))
    rng.shuffle(indices)

    val_size = int(val_split * len(x_train))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    print("Gotowe!")
    print(f"Train: {len(x_train)} przykładów")
    print(f"Val:   {len(x_val)} przykładów")
    print(f"Test:  {len(x_test)} przykładów")

    # Tworzenie TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000, seed=seed)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    train_ds, val_ds, test_ds, (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_data()
    print("Przykład:")
    print("x_train:", x_train.shape, x_train.dtype, x_train.min(), x_train.max())
    print("y_train:", y_train.shape, y_train.dtype)
    print("\nTensorFlow datasets utworzone pomyślnie!")