import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
import numpy as np


def plot_training_history(history, title, filename):
    plt.figure(figsize=(12, 5))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def evaluate_model(model, X_test, y_test_int, label_encoder, title, vis_path):
    y_pred = model.predict(X_test)
    y_pred_labels = tf.argmax(y_pred, axis=1).numpy()

    print(f"\n--- Classification Report: {title} ---")
    print(classification_report(y_test_int, y_pred_labels, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test_int, y_pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {title}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(vis_path)
    plt.close()