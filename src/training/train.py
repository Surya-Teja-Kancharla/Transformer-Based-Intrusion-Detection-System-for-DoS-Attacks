import torch
from tqdm import tqdm

def train(model, classifier, loader, optimizer, criterion, device):
    model.train()
    classifier.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = classifier(model(X))
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
