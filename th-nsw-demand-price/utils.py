import numpy as np
import torch
from torch.utils.data import Dataset

def create_sequences(X, y, seq_length=336, pred_length=48):
    Xs, ys = [], []
    for i in range(len(X) - seq_length - pred_length + 1):
        Xs.append(X[i : i + seq_length])
        ys.append(y[i + seq_length : i + seq_length + pred_length])   # shape: (pred_length, n_targets)
    return np.array(Xs), np.array(ys)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]