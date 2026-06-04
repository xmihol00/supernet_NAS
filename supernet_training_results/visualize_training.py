import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent
metrics = json.loads((OUT_DIR / "metrics.json").read_text())

epochs     = [e["epoch"]              for e in metrics]
train_loss = [e["train"]["loss"]      for e in metrics]
train_acc1 = [e["train"]["acc1"]      for e in metrics]
train_acc5 = [e["train"]["acc5"]      for e in metrics]
val_max_loss = [e["val_max"]["loss"]  for e in metrics]
val_max_acc1 = [e["val_max"]["acc1"]  for e in metrics]
val_max_acc5 = [e["val_max"]["acc5"]  for e in metrics]
val_min_loss = [e["val_min"]["loss"]  for e in metrics]
val_min_acc1 = [e["val_min"]["acc1"]  for e in metrics]
val_min_acc5 = [e["val_min"]["acc5"]  for e in metrics]

best_epoch = int(np.argmax(val_max_acc1))
best_acc1  = val_max_acc1[best_epoch]

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "Supernet Training Progress — ImageNet (82 epochs)\n"
    "Sandwich rule + Inplace distillation · lr=0.1 · batch size=96",
    fontsize=13, fontweight="bold", y=0.98,
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

# ── colour palette ──────────────────────────────────────────────────────────
C_TRAIN   = "#2196F3"   # blue
C_MAX     = "#4CAF50"   # green  (max subnet)
C_MIN     = "#FF9800"   # orange (min subnet)
ALPHA_SHD = 0.12

def _add_best(ax, x, y, label_y_offset=2.5):
    ax.axvline(best_epoch, color="red", lw=0.9, ls="--", alpha=0.6)
    ax.annotate(
        f"best ep {best_epoch}\n{best_acc1:.1f}%",
        xy=(best_epoch, y[best_epoch]),
        xytext=(best_epoch + 2, y[best_epoch] + label_y_offset),
        fontsize=7, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
    )

# ── 1. Training loss ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, train_loss, color=C_TRAIN, lw=1.8, label="Train loss")
ax1.set_title("Training Loss", fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-entropy loss")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── 2. Validation loss ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, val_max_loss, color=C_MAX, lw=1.8, label="Max subnet")
ax2.plot(epochs, val_min_loss, color=C_MIN, lw=1.8, label="Min subnet")
ax2.fill_between(epochs, val_min_loss, val_max_loss, alpha=ALPHA_SHD, color="grey")
ax2.set_title("Validation Loss (max / min subnet)", fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Cross-entropy loss")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── 3. Train vs Val top-1 accuracy ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(epochs, train_acc1, color=C_TRAIN, lw=1.8, label="Train top-1")
ax3.plot(epochs, val_max_acc1, color=C_MAX, lw=1.8, label="Val max top-1")
ax3.plot(epochs, val_min_acc1, color=C_MIN, lw=1.8, label="Val min top-1")
ax3.fill_between(epochs, val_min_acc1, val_max_acc1, alpha=ALPHA_SHD, color="grey")
ax3.scatter([best_epoch], [best_acc1], color="red", zorder=5, s=50)
ax3.annotate(
    f" best {best_acc1:.2f}%\n ep {best_epoch}",
    xy=(best_epoch, best_acc1), fontsize=7, color="red",
    xytext=(best_epoch + 2, best_acc1 - 5),
    arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
)
ax3.set_title("Top-1 Accuracy", fontweight="bold")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Accuracy (%)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── 4. Train vs Val top-5 accuracy ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(epochs, train_acc5, color=C_TRAIN, lw=1.8, label="Train top-5")
ax4.plot(epochs, val_max_acc5, color=C_MAX, lw=1.8, label="Val max top-5")
ax4.plot(epochs, val_min_acc5, color=C_MIN, lw=1.8, label="Val min top-5")
ax4.fill_between(epochs, val_min_acc5, val_max_acc5, alpha=ALPHA_SHD, color="grey")
ax4.set_title("Top-5 Accuracy", fontweight="bold")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Accuracy (%)")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ── 5. Accuracy gap (max - min subnet) ───────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
gap1 = [m - n for m, n in zip(val_max_acc1, val_min_acc1)]
gap5 = [m - n for m, n in zip(val_max_acc5, val_min_acc5)]
ax5.plot(epochs, gap1, color="#9C27B0", lw=1.8, label="Top-1 gap")
ax5.plot(epochs, gap5, color="#607D8B", lw=1.8, ls="--", label="Top-5 gap")
ax5.set_title("Accuracy Gap (max − min subnet)", fontweight="bold")
ax5.set_xlabel("Epoch")
ax5.set_ylabel("Δ Accuracy (%)")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# ── 6. Summary table ─────────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")

last = metrics[-1]
rows = [
    ["Metric", "Max subnet", "Min subnet"],
    ["Best val top-1", f"{best_acc1:.2f}% (ep {best_epoch})", "—"],
    ["Final val top-1", f"{val_max_acc1[-1]:.2f}%", f"{val_min_acc1[-1]:.2f}%"],
    ["Final val top-5", f"{val_max_acc5[-1]:.2f}%", f"{val_min_acc5[-1]:.2f}%"],
    ["Final val loss",  f"{val_max_loss[-1]:.4f}",  f"{val_min_loss[-1]:.4f}"],
    ["Final train top-1", f"{train_acc1[-1]:.2f}%", "—"],
    ["Params",  "10.35 M",  "1.58 M"],
    ["Memory",  "13.38 MB", "3.85 MB"],
]
table = ax6.table(
    cellText=rows[1:],
    colLabels=rows[0],
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(8)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#455A64")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#ECEFF1")
    cell.set_edgecolor("#B0BEC5")
ax6.set_title("Key Metrics Summary", fontweight="bold", pad=4)

out_path = OUT_DIR / "training_progress.png"
fig.savefig(out_path, dpi=400, bbox_inches="tight")
print(f"Saved → {out_path}")
