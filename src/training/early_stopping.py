# src/training/early_stopping.py

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_f1):
        if self.best_f1 is None:
            self.best_f1 = current_f1
            return

        if current_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = current_f1
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
