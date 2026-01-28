import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_metrics(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = classifier(model(X)).argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return acc, p, r, f1
