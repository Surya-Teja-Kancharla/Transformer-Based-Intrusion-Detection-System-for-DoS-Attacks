import os
import sys
import yaml
import torch
import numpy as np
import time

from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
from data.preprocessing import load_and_clean, normalize, correlation_prune, quantile_clip, encode_labels
from data.windowing import create_windows
from data.dataset import IDSWindowDataset
from data.split_save import split_and_save
from data.load_processed import load_processed

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
from models.transformer import LightweightTransformer
from models.classifier import Classifier

# ------------------------------------------------------------------
# Training & utils
# ------------------------------------------------------------------
from training.train import train
from training.early_stopping import EarlyStopping
from training.evaluate import evaluate_metrics
from utils.logger import get_logger
from utils.seed import set_seed
from utils.results import save_results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # ------------------------------------------------------------------
    # Load config & setup
    # ------------------------------------------------------------------
    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "configs", "config.yaml")))
    set_seed(cfg["dataset"]["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = get_logger(
        log_dir=os.path.join(PROJECT_ROOT, "logs"),
        name="lightweight_transformer_ids"
    )
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load & preprocess raw data
    # ------------------------------------------------------------------
    df = load_and_clean(os.path.join(PROJECT_ROOT, cfg["dataset"]["raw_path"]))
    X, y_raw = normalize(df)

    # Encode string labels -> integer class IDs
    # Keep only Benign + DoS attacks (remove Heartbleed)
    allowed_classes = [
        "BENIGN",
        "DoS GoldenEye",
        "DoS Hulk",
        "DoS Slowhttptest",
        "DoS slowloris"
    ]

    labels_filtered, mask = encode_labels(
        y_raw.values,
        allowed_classes=allowed_classes
    )

    X = X[mask]  # keep features aligned
    y_raw = labels_filtered

    # Encode filtered labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    logger.info(f"Final label mapping: {dict(enumerate(le.classes_))}")

    X = correlation_prune(X, cfg["preprocessing"]["correlation_threshold"])
    q_low, q_high = cfg["preprocessing"]["quantile_clip"]

    X = quantile_clip(
        X,
        q_low,
        q_high
    )


    # ------------------------------------------------------------------
    # Burst window sequencing
    # ------------------------------------------------------------------
    X_seq, y_seq = create_windows(
        X,
        y,
        cfg["windowing"]["window_size"],
        cfg["windowing"]["stride"]
    )

    # ------------------------------------------------------------------
    # Persistent train / val / test split
    # ------------------------------------------------------------------
    processed_path = os.path.join(PROJECT_ROOT, cfg["dataset"]["processed_path"])

    split_and_save(
        X_seq,
        y_seq,
        processed_path,
        test_size=cfg["dataset"]["test_size"],
        val_size=cfg["dataset"]["val_size"],
        seed=cfg["dataset"]["random_seed"]
    )

    (train_X, train_y), (val_X, val_y), (test_X, test_y) = load_processed(processed_path)

    train_ds = IDSWindowDataset(train_X, train_y)
    val_ds   = IDSWindowDataset(val_X, val_y)
    test_ds  = IDSWindowDataset(test_X, test_y)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # Separate transformer config from classifier config
    transformer_cfg = {
        k: v for k, v in cfg["model"].items()
        if k != "num_classes"
    }

    model = LightweightTransformer(
        input_dim=train_X.shape[2],
        **transformer_cfg
    ).to(device)

    classifier = Classifier(
        cfg["model"]["d_model"],
        cfg["model"]["num_classes"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        verbose=True
    )

    # ------------------------------------------------------------------
    # Parameter count (paper-critical)
    # ------------------------------------------------------------------
    total_params = count_parameters(model) + count_parameters(classifier)
    logger.info(f"Total trainable parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Class imbalance handling (TRAIN ONLY)
    # ------------------------------------------------------------------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_y.numpy()),
        y=train_y.numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    early_stopper = EarlyStopping(patience=5)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train(
            model,
            classifier,
            train_loader,
            optimizer,
            criterion,
            device
        )

        acc, p, r, f1 = evaluate_metrics(
            model,
            classifier,
            val_loader,
            device
        )

        logger.info(
            f"Epoch {epoch+1:03d} | "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Prec: {p:.4f} | "
            f"Recall: {r:.4f} | "
            f"F1: {f1:.4f}"
        )

        scheduler.step(f1)
        early_stopper.step(f1)

        if early_stopper.should_stop:
            logger.info("Early stopping triggered.")
            break

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    acc, p, r, f1 = evaluate_metrics(
        model,
        classifier,
        test_loader,
        device
    )

    logger.info(
        f"TEST RESULTS | "
        f"Acc: {acc:.4f} | "
        f"Prec: {p:.4f} | "
        f"Recall: {r:.4f} | "
        f"F1: {f1:.4f}"
    )

    # ------------------------------------------------------------------
    # Inference latency (paper-critical)
    # ------------------------------------------------------------------
    model.eval()
    classifier.eval()

    start = time.time()
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            _ = classifier(model(X))
    end = time.time()

    avg_latency = (end - start) / len(test_loader)
    logger.info(f"Average inference latency per batch: {avg_latency:.6f} seconds")

    # ------------------------------------------------------------------
    # Confusion matrix (paper-ready)
    # ------------------------------------------------------------------
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = classifier(model(X)).argmax(dim=1).cpu()
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    cm = confusion_matrix(y_true, y_pred)

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    # ------------------------------------------------------------------
    # Save paper-ready results
    # ------------------------------------------------------------------
    metrics = {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "parameters": total_params,
        "latency": avg_latency
    }

    save_results(metrics, results_dir)

    logger.info("Experiment completed successfully.")


if __name__ == "__main__":
    main()
