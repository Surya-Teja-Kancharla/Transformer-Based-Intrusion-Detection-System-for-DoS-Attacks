# src/utils/results.py

import json
import os

def save_results(metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # JSON (for reproducibility)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # LaTeX table (for paper)
    latex = r"""
        \begin{table}[h]
        \centering
        \caption{Performance of Lightweight Transformer IDS}
        \begin{tabular}{lcccc}
        \hline
        Metric & Accuracy & Precision & Recall & F1-score \\
        \hline
        Proposed Model & %.4f & %.4f & %.4f & %.4f \\
        \hline
        \end{tabular}
        \end{table}
        """ % (
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"]
            )

    with open(os.path.join(save_dir, "results.tex"), "w") as f:
        f.write(latex)
