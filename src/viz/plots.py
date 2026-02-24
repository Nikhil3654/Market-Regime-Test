from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_equity_curve(dates, equity: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(dates, equity)
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("equity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.xticks([0, 1], ["down", "up"])
    plt.yticks([0, 1], ["down", "up"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()