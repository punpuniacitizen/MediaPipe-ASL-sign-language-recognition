"""Train the ASL letter classifier from stored hand landmarks.

Differences from the original version, and why:

  * Input comes from `landmarks.npz`, not from a folder of images. Each sample is
    rendered on the fly by `preprocessing.render_skeleton()` — the same function the
    live translator calls — so the training distribution matches inference by
    construction rather than by coincidence.
  * The train/validation split is a contiguous block per class, not a random draw.
    These datasets are sequences of near-identical frames from the same recording, so
    a random split puts a near-duplicate of almost every validation image into the
    training set. That is what inflated the old 97.9%.
  * Augmentation happens in landmark space (rotation, scale, translation, jitter,
    mirroring), which leaves stroke weight and palette untouched, so every augmented
    sample still looks like something the renderer could produce from a real hand.
  * BatchNorm, Dropout and global average pooling replace the bare Flatten -> Dense
    stack that memorised the old training set.

Usage:
    python train_model.py --landmarks landmarks.npz --epochs 40
    python train_model.py --landmarks landmarks.npz --test-dataset ayuraj
"""

import argparse
import json
import os
import re
import threading

import numpy as np

import preprocessing as pp

_local = threading.local()


def _rng(base_seed):
    """Per-thread generator: tf.data maps this across several worker threads."""
    if not hasattr(_local, "rng"):
        _local.rng = np.random.default_rng(base_seed + threading.get_ident() % 100_000)
    return _local.rng


def configure_tensorflow(use_mixed_precision):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        # Without this TensorFlow reserves the whole card up front, which on a 6 GB
        # laptop GPU leaves nothing for the desktop.
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        print(f"GPU detected: {[g.name for g in gpus]}")
    else:
        print("WARNING: no GPU detected; training will run on the CPU.")
        print("         On Linux: pip install 'tensorflow[and-cuda]'")

    if use_mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision on (the output layer stays float32).")

    return tf


def natural_key(path):
    """Sort A2.jpg before A10.jpg.

    Each token becomes a (kind, number, text) triple so a filename that starts with a
    digit never gets compared against one that starts with a letter.
    """
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    return tuple(
        (0, int(token), "") if token.isdigit() else (1, 0, token.lower())
        for token in re.split(r"(\d+)", stem) if token
    )


def build_splits(data, val_fraction, test_dataset):
    """Contiguous per-class blocks, so validation frames are not neighbours of training ones."""
    labels = data["labels"]
    datasets = data["datasets"]
    paths = data["paths"]
    dataset_names = [str(n) for n in data["dataset_names"]]

    test_id = None
    if test_dataset:
        if test_dataset not in dataset_names:
            raise SystemExit(
                f"Error: '{test_dataset}' is not in the npz. Available: {dataset_names}"
            )
        test_id = dataset_names.index(test_dataset)

    train_idx, val_idx, test_idx = [], [], []

    for dataset_id in np.unique(datasets):
        for label in np.unique(labels):
            group = np.where((datasets == dataset_id) & (labels == label))[0]
            if group.size == 0:
                continue
            # Plain `sorted`, not np.argsort: the keys are tuples, and NumPy would turn
            # a list of them into a 2-D array and sort along the wrong axis.
            group = np.array(sorted(group.tolist(), key=lambda i: natural_key(paths[i])))

            if test_id is not None and dataset_id == test_id:
                test_idx.extend(group.tolist())
                continue

            cut = int(round(len(group) * (1.0 - val_fraction)))
            cut = min(max(cut, 1), len(group) - 1) if len(group) > 1 else len(group)
            train_idx.extend(group[:cut].tolist())
            val_idx.extend(group[cut:].tolist())

    return np.array(train_idx), np.array(val_idx), np.array(test_idx, dtype=int)


def make_dataset(tf, points, labels, size, batch_size, training, seed):
    def render(pts):
        if training:
            pts = pp.augment_landmarks(pts, _rng(seed))
        return pp.render_skeleton(pts, size=size)

    ds = tf.data.Dataset.from_tensor_slices((points, labels))
    if training:
        ds = ds.shuffle(min(len(points), 20_000), seed=seed, reshuffle_each_iteration=True)

    def to_image(pts, label):
        image = tf.numpy_function(render, [pts], tf.uint8, name="render_skeleton")
        image.set_shape((size, size, 3))
        return image, label

    ds = ds.map(to_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Keras defaults BatchNorm's moving statistics to momentum 0.99, which assumes a long
# run: after a few hundred steps they are still part-way to the real values, and with
# seven normalisation layers stacked the error compounds until the model predicts a
# single class for everything at inference. That collapse also poisons EarlyStopping,
# which then treats epoch 1 as the best. 0.9 converges within ~100 steps and makes short
# runs behave like long ones.
BN_MOMENTUM = 0.9


def build_model(tf, input_size, num_classes):
    layers = tf.keras.layers

    def block(x, filters, name):
        for i in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
            # The activation visualiser reads the first block's ReLU by name, so it
            # must not depend on autogenerated tensor names surviving a retrain.
            x = layers.ReLU(name=f"{name}_relu" if i == 0 else None)(x)
        x = layers.MaxPooling2D()(x)
        return layers.Dropout(0.25)(x)

    inputs = layers.Input((input_size, input_size, 3), name="input_image")
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = block(x, 32, "conv1")
    x = block(x, 64, "conv2")
    x = block(x, 128, "conv3")

    # Global average pooling instead of Flatten(): roughly ten times fewer parameters
    # into the head, which is where the old model did its memorising.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = layers.ReLU(name="dense_features")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, dtype="float32", name="logits")(x)

    return tf.keras.Model(inputs, outputs, name="asl_cnn")


def plot_history(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["accuracy"], label="Training")
    ax1.plot(epochs, history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["loss"], label="Training")
    ax2.plot(epochs, history["val_loss"], label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Training curves -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--landmarks", default="landmarks.npz")
    parser.add_argument("--out", default="asl_cnn_model.keras")
    parser.add_argument("--input-size", type=int, default=pp.MODEL_INPUT_SIZE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-dataset", default="",
                        help="Name of a dataset to hold out entirely as an unseen test set")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="Rarely helps here: the bottleneck is rendering, not the GPU.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.landmarks):
        raise SystemExit(f"Error: '{args.landmarks}' not found. Run prepare_dataset.py first.")

    data = np.load(args.landmarks, allow_pickle=False)
    class_names = [str(c) for c in data["class_names"]]
    points = data["points"].astype(np.float32)
    labels = data["labels"].astype(np.int32)

    train_idx, val_idx, test_idx = build_splits(data, args.val_fraction, args.test_dataset)

    print(f"Classes ({len(class_names)}): {' '.join(class_names)}")
    print(f"Train {len(train_idx):,}  |  validation {len(val_idx):,}  |  held-out test {len(test_idx):,}")

    counts = np.bincount(labels[train_idx], minlength=len(class_names))
    print(f"Training samples per class: min {counts.min():,}  max {counts.max():,}  "
          f"imbalance {counts.max() / max(counts.min(), 1):.1f}:1")

    tf = configure_tensorflow(args.mixed_precision)
    tf.keras.utils.set_random_seed(args.seed)

    train_ds = make_dataset(tf, points[train_idx], labels[train_idx], args.input_size,
                            args.batch_size, True, args.seed)
    val_ds = make_dataset(tf, points[val_idx], labels[val_idx], args.input_size,
                          args.batch_size, False, args.seed)

    model = build_model(tf, args.input_size, len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

    total = counts.sum()
    class_weight = {
        i: float(total / (len(class_names) * c)) if c else 1.0 for i, c in enumerate(counts)
    }

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(args.out, monitor="val_accuracy",
                                           save_best_only=True, verbose=0),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    model.save(args.out)
    print(f"\nModel saved -> {args.out}")

    with open("class_names.txt", "w") as f:
        f.write(",".join(class_names))
    print(f"Classes saved -> class_names.txt ({len(class_names)})")

    np.savez(
        "split.npz",
        train=train_idx, val=val_idx, test=test_idx,
        landmarks=np.array(args.landmarks), input_size=np.array(args.input_size),
    )
    print("Split indices -> split.npz (evaluate.py reuses exactly this cut)")

    with open("training_history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # Written straight into docs/ so the curve the README shows is always the one from
    # the most recent run, rather than a copy someone remembered to update.
    os.makedirs("docs", exist_ok=True)
    plot_history(history.history, os.path.join("docs", "training-accuracy.png"))

    best = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {best:.2%}")
    print("Run evaluate.py for the confusion matrix and the unseen-subject number.")


if __name__ == "__main__":
    main()
