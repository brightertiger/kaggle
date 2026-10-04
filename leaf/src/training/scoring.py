import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def calculate_accuracy(predictions, targets):
    return accuracy_score(targets, np.argmax(predictions, axis=1))


def generate_classification_report(predictions, targets, class_names=None):
    return classification_report(targets, np.argmax(predictions, axis=1),
                                 labels=np.arange(predictions.shape[1]),
                                 target_names=class_names, zero_division=0)


def plot_confusion_matrix(predictions, targets, class_names=None, save_path=None):
    # Plotting is optional at runtime and does not initialize a GUI during imports.
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(targets, np.argmax(predictions, axis=1),
                          labels=np.arange(predictions.shape[1]), normalize='true')
    fig, ax = plt.subplots(figsize=(10, 8))
    ticks = class_names if class_names is not None else 'auto'
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=ticks, yticklabels=ticks, ax=ax)
    ax.set(title='Confusion Matrix', xlabel='Predicted', ylabel='Actual')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def evaluate_model_performance(predictions, targets, class_names=None, save_path=None):
    accuracy = calculate_accuracy(predictions, targets)
    report = generate_classification_report(predictions, targets, class_names)
    print(f'Model Accuracy: {accuracy:.4f}\n\nClassification Report:\n{report}')
    plot_confusion_matrix(predictions, targets, class_names, save_path)
    return accuracy, report
