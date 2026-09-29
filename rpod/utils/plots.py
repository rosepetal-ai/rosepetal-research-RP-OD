"""Validation plots: precision-recall curves, confusion matrix, and a grid of images with real and predicted boxes."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from ..engine.predictor import class_color  # noqa: E402

GRID_IMAGES = 16
TILE = 480


def plot_pr_curves(metrics, names, path):
    """Precision-recall at IoU 0.5 for each class, with the operating point of the recommended threshold."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    thr = metrics.threshold
    for c, (scores, tp, n_gt) in enumerate(metrics.curves['pr50']):
        if n_gt == 0:
            continue
        col = np.array(class_color(c)) / 255
        tp_cum, det = np.cumsum(tp), np.arange(1, len(tp) + 1)
        recall = np.concatenate([[0], tp_cum / n_gt])
        precision = np.concatenate([[1], tp_cum / det])
        ax.plot(recall, precision, color=col, lw=1.6, label=f"{names[c]}  AP50 {metrics.per_class[c]['map50']:.3f}")
        r = metrics.per_class[c]
        ax.plot(r['recall'], r['precision'], 'o', color=col, ms=7, mec='black')
    ax.set(xlim=(0, 1.01), ylim=(0, 1.01), xlabel='recall', ylabel='precision',
           title=f'Precision-recall (IoU 0.5) · dots: threshold {thr:.3f}')
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='lower left', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'no objects in this split', ha='center', va='center', color='gray')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(metrics, names, path):
    m = metrics.confusion_matrix
    labels = [names[i] for i in range(len(names))] + ['background']
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(6 + len(labels), 4 + 0.6 * len(labels)), dpi=120)
    for ax, data, title, fmt in ((axes[0], m, 'counts', '{:d}'), (axes[1], norm, 'per true class', '{:.2f}')):
        ax.imshow(data, cmap='Blues')
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xlabel='predicted', ylabel='true',
               title=f'Confusion matrix, {title}\nthreshold {metrics.threshold:.3f}')
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        top = data.max() if data.size else 1
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = data[i, j]
                ax.text(j, i, fmt.format(int(v) if fmt == '{:d}' else v), ha='center', va='center', fontsize=8,
                        color='white' if v > top / 2 else 'black')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_image_grid(items, names, threshold, path):
    """items: up to 16 (rgb image, gt boxes xyxy, gt classes, pred boxes, pred scores, pred classes).

    Real boxes in white, predictions above the threshold in their class color with name and score."""
    tiles = []
    for img, gb, gc, pb, ps, pc in items[:GRID_IMAGES]:
        im = Image.fromarray(np.ascontiguousarray(img))
        s = TILE / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))))
        d = ImageDraw.Draw(im)
        for b in gb * s:
            d.rectangle(b.tolist(), outline=(255, 255, 255), width=2)
        for b, sc, c in zip(pb * s, ps, pc):
            if sc > threshold:
                col = class_color(int(c))
                d.rectangle(b.tolist(), outline=col, width=2)
                text = f'{names[int(c)]} {sc:.2f}'
                x0, y0, x1, y1 = d.textbbox((0, 0), text)
                ty = b[1] - (y1 - y0) - 4 if b[1] > (y1 - y0) + 4 else b[1]   # above the box, inside if no room
                d.rectangle([b[0], ty, b[0] + x1 + 4, ty + y1 + 4], fill=col)
                d.text((b[0] + 2, ty + 2), text, fill=(0, 0, 0))
        canvas = Image.new('RGB', (TILE, TILE), (24, 24, 24))
        canvas.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
        tiles.append(canvas)
    if not tiles:
        return
    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / cols))
    grid = Image.new('RGB', (cols * TILE, rows * TILE), (0, 0, 0))
    for k, t in enumerate(tiles):
        grid.paste(t, ((k % cols) * TILE, (k // cols) * TILE))
    grid.save(path, quality=90)


def save_plots(metrics, names, items, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_pr_curves(metrics, names, out_dir / 'pr_curves.png')
    plot_confusion_matrix(metrics, names, out_dir / 'confusion_matrix.png')
    plot_image_grid(items, names, metrics.threshold, out_dir / 'predictions.jpg')
