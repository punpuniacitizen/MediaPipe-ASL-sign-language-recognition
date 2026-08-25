"""Per-class metrics and confusion matrix on the exact split train_model.py used.

The old pipeline reported a single validation accuracy taken from a random split over
near-duplicate frames, which is why it read 97.9% while the webcam struggled. This
script reports two separate numbers and keeps them clearly apart:

  * **validacion** — held-out contiguous block from the same recordings. Measures how
    well the model learned these hands.
  * **test** — an entire dataset the model never saw, ideally recorded by different
    people. This is the number that predicts live behaviour, and it is normally a lot
    lower. Treat it as the honest one.

Two further readings go beyond plain accuracy, because plain accuracy describes a bare
argmax and the translator never uses a bare argmax:

  * **threshold precision and coverage** — the translator only appends a letter when the
    smoothed confidence clears COMMIT_CONFIDENCE. What matters in use is how often it
    commits at all (coverage) and how often a commit is right (precision). A model with
    modest accuracy but high precision at the threshold feels dependable; a
    poorly-calibrated one with better accuracy feels erratic.
  * **jitter stability** — the original symptom was the prediction flipping while the
    hand was held still. Re-rendering the same landmarks under small perturbations and
    counting how often the answer changes measures that symptom directly.

Usage:
    python evaluate.py                        # usa split.npz
    python evaluate.py --model asl_cnn_model.keras --split split.npz
"""

import argparse
import os

import numpy as np

import preprocessing as pp

# Mirrored from realtime_translator.py, which is the source of truth. They live there
# because that is where they are applied; they are repeated here rather than imported
# because importing the translator would drag in mediapipe and onnxruntime, which do not
# exist in the training environment. Keep the two copies in step.
CONFIDENCE_FLOOR = 0.50
COMMIT_CONFIDENCE = 0.75


def render_batch(points, size):
    return np.stack([pp.render_skeleton(p, size=size) for p in points]).astype(np.float32)


def predict(model, points, size, batch_size=256):
    out = []
    for start in range(0, len(points), batch_size):
        chunk = render_batch(points[start:start + batch_size], size)
        out.append(model.predict(chunk, verbose=0))
    return np.concatenate(out) if out else np.zeros((0, 0))


def softmax(logits):
    """The model emits raw logits; the translator applies this before thresholding."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def jitter_landmarks(points, rng, degrees=3.0, shift=0.004, noise=0.004):
    """Perturb landmarks the way a *still* hand drifts between frames.

    Far gentler than `augment_landmarks`: this is not augmentation but a model of
    MediaPipe's frame-to-frame noise once LANDMARK_ALPHA smoothing has damped it. Nothing
    is clipped afterwards — a hand occupies HAND_FILL of the canvas, leaving ample margin
    at this magnitude, and clipping per joint would bend fingers rather than move the
    hand.
    """
    pts = np.asarray(points, dtype=np.float32).copy()

    centre = np.float32([0.5, 0.5])
    angle = np.deg2rad(rng.uniform(-degrees, degrees))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    pts = (pts - centre) @ np.float32([[cos_a, -sin_a], [sin_a, cos_a]]).T + centre

    pts += rng.uniform(-shift, shift, size=2).astype(np.float32)
    pts += rng.normal(0.0, noise, size=pts.shape).astype(np.float32)
    return pts.astype(np.float32)


def threshold_report(probabilities, labels):
    """How the model behaves once the translator's confidence gates are applied."""
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == labels

    print("\nBehaviour at the translator's confidence gates:")
    print(f"  {'threshold':<26}{'coverage':>12}{'precision':>12}{'wrong commits':>16}")
    for name, threshold in (("CONFIDENCE_FLOOR", CONFIDENCE_FLOOR),
                            ("COMMIT_CONFIDENCE", COMMIT_CONFIDENCE)):
        passing = confidence >= threshold
        coverage = float(passing.mean())
        label = f"{name} >= {threshold:.2f}"
        if passing.any():
            precision = float(correct[passing].mean())
            wrong = f"{int((~correct[passing]).sum()):,} / {int(passing.sum()):,}"
            print(f"  {label:<26}{coverage:>11.1%}{precision:>12.1%}{wrong:>16}")
        else:
            print(f"  {label:<26}{coverage:>11.1%}{'-':>12}{'-':>16}")

    # Silence is better than a confident mistake here: an unconfirmed letter just makes
    # the user hold the sign a moment longer, whereas a wrong commit lands in the buffer.
    print("  Coverage is how often a letter is offered at all; precision is how often")
    print("  the offered letter is right. Low coverage with high precision is workable.")


def stability_report(model, points, labels, size, trials, rng):
    """Fraction of still-hand renders whose prediction survives small perturbations."""
    if len(points) == 0:
        return

    baseline = predict(model, points, size).argmax(axis=1)

    agreements = np.zeros(len(points), dtype=np.int32)
    for _ in range(trials):
        jittered = np.stack([jitter_landmarks(p, rng) for p in points])
        agreements += (predict(model, jittered, size).argmax(axis=1) == baseline)

    per_sample = agreements / trials
    rock_solid = float((agreements == trials).mean())

    print(f"\nJitter stability ({len(points):,} samples x {trials} perturbations):")
    print(f"  predictions unchanged across every perturbation : {rock_solid:.1%} of hands")
    print(f"  mean agreement with the unjittered prediction   : {per_sample.mean():.1%}")

    # A hand the model is unsure about is exactly the one that flickers on screen, so
    # this splits the flicker by whether the baseline answer was even right.
    correct_baseline = baseline == labels
    if correct_baseline.any():
        print(f"  agreement where the baseline was correct        : "
              f"{per_sample[correct_baseline].mean():.1%}")
    if (~correct_baseline).any():
        print(f"  agreement where the baseline was wrong          : "
              f"{per_sample[~correct_baseline].mean():.1%}")


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

    threshold_report(softmax(logits), labels)

    return accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="asl_cnn_model.keras")
    parser.add_argument("--split", default="split.npz")
    parser.add_argument("--landmarks", default="")
    parser.add_argument("--stability-trials", type=int, default=8,
                        help="Perturbations per hand; 0 skips the stability check")
    parser.add_argument("--stability-samples", type=int, default=1500,
                        help="Hands to perturb, sampled from the test split")
    parser.add_argument("--seed", type=int, default=123)
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

    # Run on the unseen split: stability on hands the model already knows would flatter
    # it, and the live camera never shows it a hand it has seen before.
    if args.stability_trials > 0:
        stability_idx = test_idx if len(test_idx) else val_idx
        rng = np.random.default_rng(args.seed)
        if len(stability_idx) > args.stability_samples:
            stability_idx = rng.choice(stability_idx, args.stability_samples, replace=False)
        stability_report(model, points[stability_idx], labels[stability_idx], size,
                         args.stability_trials, rng)

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
