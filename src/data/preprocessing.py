import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in df.columns if 'IP' in c or 'Port' in c], errors='ignore')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def normalize(df):
    scaler = StandardScaler()
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    features = scaler.fit_transform(features)
    return features, labels

def correlation_prune(X, threshold=0.9):
    std = X.std(axis=0)
    non_zero_var_idx = np.where(std > 0)[0]
    X = X[:, non_zero_var_idx]

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr)

    upper = np.triu(np.abs(corr), 1)
    remove = np.unique(np.where(upper > threshold)[1])

    return np.delete(X, remove, axis=1)

def quantile_clip(X, q_low=0.01, q_high=0.99):
    low = np.quantile(X, q_low, axis=0)
    high = np.quantile(X, q_high, axis=0)
    return np.clip(X, low, high)

def encode_labels(labels, allowed_classes=None):
    """
    Encode string labels into integer class IDs.
    Optionally filter to allowed classes only.
    """
    labels = np.array(labels)

    if allowed_classes is not None:
        mask = np.isin(labels, allowed_classes)
        labels = labels[mask]
        return labels, mask

    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    return encoded, le

