import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_accuracy(predictions, targets):
    pred_labels = np.argmax(predictions, axis=1)
    return accuracy_score(targets, pred_labels)

def generate_classification_report(predictions, targets, class_names=None):
    pred_labels = np.argmax(predictions, axis=1)
    return classification_report(targets, pred_labels, target_names=class_names)

def plot_confusion_matrix(predictions, targets, class_names=None, save_path=None):
    pred_labels = np.argmax(predictions, axis=1)
    cm = confusion_matrix(targets, pred_labels, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def evaluate_model_performance(predictions, targets, class_names=None):
    accuracy = calculate_accuracy(predictions, targets)
    report = generate_classification_report(predictions, targets, class_names)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    plot_confusion_matrix(predictions, targets, class_names)
    
    return accuracy, report
