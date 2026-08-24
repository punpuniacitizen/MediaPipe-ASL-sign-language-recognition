"""Per-class metrics and confusion matrix on the exact split train_model.py used.

The old pipeline reported a single validation accuracy taken from a random split over
near-duplicate frames, which is why it read 97.9% while the webcam struggled. This
script reports two separate numbers and keeps them clearly apart:

  * **validacion** — held-out contiguous block from the same recordings. Measures how
    well the model learned these hands.
  * **test** — an entire dataset the model never saw, ideally recorded by different
    people. This is the number that predicts live behaviour, and it is normally a lot
    lower. Treat it as the honest one.

Usage:
    python evaluate.py                        # usa split.npz
    python evaluate.py --model asl_cnn_model.keras --split split.npz
"""

import argparse
import os

import numpy as np

import preprocessing as pp


def render_batch(points, size):
    return np.stack([pp.render_skeleton(p, size=size) for p in points]).astype(np.float32)


def predict(model, points, size, batch_size=256):
    out = []
    for start in range(0, len(points), batch_size):
        chunk = render_batch(points[start:start + batch_size], size)
        out.append(model.predict(chunk, verbose=0))
    return np.concatenate(out) if out else np.zeros((0, 0))


def plot_confusion(matrix, class_names, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normed = matrix.astype(float) / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(normed, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, fontsize=8)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Confusion matrix -> {path}")


def report(name, model, points, labels, class_names, size, out_image):
    from sklearn.metrics import classification_report, confusion_matrix

    if len(points) == 0:
        print(f"\n[{name}] empty, skipping.")
        return None

    logits = predict(model, points, size)
    predicted = logits.argmax(axis=1)
    accuracy = float((predicted == labels).mean())

    print(f"\n{'=' * 62}\n[{name}]  {len(points):,} samples  |  accuracy {accuracy:.2%}\n{'=' * 62}")
    print(classification_report(labels, predicted, labels=range(len(class_names)),
                                target_names=class_names, zero_division=0, digits=3))

    matrix = confusion_matrix(labels, predicted, labels=range(len(class_names)))
    plot_confusion(matrix, class_names, out_image, f"Confusion matrix — {name}")

    # Which pairs the model actually mixes up. Letters that share a handshape (M/N/S/T
    # are all a fist with the thumb in different places) are expected here; anything
    # else is worth a look.
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and matrix[i, j]:
                confusions.append((matrix[i, j] / max(matrix[i].sum(), 1), class_names[i], class_names[j], matrix[i, j]))
    confusions.sort(reverse=True)
    if confusions:
        print("Most frequent confusions (true -> predicted):")
        for rate, real, pred, count in confusions[:12]:
            print(f"  {real} -> {pred}: {count:4d}  ({rate:.1%} of all '{real}')")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="asl_cnn_model.keras")
    parser.add_argument("--split", default="split.npz")
    parser.add_argument("--landmarks", default="")
    args = parser.parse_args()

    for path in (args.model, args.split):
        if not os.path.exists(path):
            raise SystemExit(f"Error: '{path}' not found. Run train_model.py first.")

    split = np.load(args.split, allow_pickle=False)
    landmarks_path = args.landmarks or str(split["landmarks"])
    if not os.path.exists(landmarks_path):
        raise SystemExit(f"Error: '{landmarks_path}' not found.")

    data = np.load(landmarks_path, allow_pickle=False)
    class_names = [str(c) for c in data["class_names"]]
    points = data["points"].astype(np.float32)
    labels = data["labels"].astype(np.int32)
    size = int(split["input_size"])

    import tensorflow as tf

    model = tf.keras.models.load_model(args.model)
    print(f"Model: {args.model}  |  input {size}x{size}  |  {len(class_names)} classes")

    val_idx, test_idx = split["val"], split["test"]

    val_acc = report("validation (same recordings)", model, points[val_idx], labels[val_idx],
                     class_names, size, "docs/confusion-validation.png")
    test_acc = report("test (unseen dataset)", model, points[test_idx], labels[test_idx],
                      class_names, size, "docs/confusion-test.png")

    print(f"\n{'=' * 62}\nSUMMARY")
    print(f"  validation : {val_acc:.2%}" if val_acc is not None else "  validation : -")
    if test_acc is not None:
        print(f"  test       : {test_acc:.2%}   <- this is the one that predicts live behaviour")
        print(f"  gap        : {(val_acc - test_acc) * 100:.1f} points")
    else:
        print("  test       : no dataset was held out.")
        print("  Retrain with --test-dataset <name> to get an honest number.")
    print("=" * 62)


if __name__ == "__main__":
    main()
