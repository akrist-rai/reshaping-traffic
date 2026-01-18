class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, loss):
        if loss < self.best:
            self.best = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def load(self, best, counter):
        self.best = best
        self.counter = counter

    def step(self, loss):
        if loss < self.best:
            self.best = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

