import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def compute_metrics(y_true, y_pred):
    """
    Compute various evaluation metrics: accuracy, precision, recall, and F1-score.
    Args:
        y_true: Ground truth labels (list or np.array)
        y_pred: Predicted labels (list or np.array)
    Returns:
        Dictionary containing accuracy, precision, recall, and F1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def print_metrics(metrics):
    """
    Print evaluation metrics.
    Args:
        metrics: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
