"""Run MediaPipe over the original ASL photos once and store the hand landmarks.

The previous pipeline trained on skeleton images somebody else had already rendered,
which is why the model saw a different kind of picture than the webcam produces. Here
we go back to the source photographs, extract the 21 landmarks ourselves, and keep only
those. Training then renders its own images through `preprocessing.render_skeleton()` —
the same function the translator uses — so the two can never drift apart again.

The landmarks for a whole dataset weigh about 20 MB, so training reads them straight
from RAM instead of walking 78k files. That removes the disk bottleneck that made the
original run slow enough to need an on-disk cache.

Expected layout: one directory per class inside each source root.

    python prepare_dataset.py \
        --source alphabet=~/datasets/asl_alphabet_train \
        --source ayuraj=~/datasets/asl_dataset \
        --out landmarks.npz
"""

import argparse
import os
import re
import sys
from collections import Counter

import cv2
import numpy as np

import preprocessing as pp

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = [chr(c) for c in range(ord("a"), ord("z") + 1)]

_hands = None


def _init_worker(pad_ratio, min_confidence):
    global _hands
    from mediapipe.python.solutions import hands as mp_hands

    _hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=pp.HAND_MODEL_COMPLEXITY,
        min_detection_confidence=min_confidence,
    )


def _natural_key(path):
    """Sort A2.jpg before A10.jpg so the contiguous split follows capture order."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", stem)]


def _detect(path, pad_ratio):
    """Return normalised (21, 2) landmarks for one photo, or None if no hand is found."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    # These datasets are cropped tight around the hand, and MediaPipe's palm detector
    # does noticeably better with some empty space around the subject.
    if pad_ratio > 0:
        border = int(round(max(image.shape[:2]) * pad_ratio))
        image = cv2.copyMakeBorder(
            image, border, border, border, border, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    height, width = image.shape[:2]
    result = _hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not result.multi_hand_landmarks:
        return None

    points_px = pp.landmarks_to_pixels(result.multi_hand_landmarks[0], width, height)
    return pp.normalize_landmarks(points_px)


def _process(job):
    path, label, dataset_id, pad_ratio = job
    points = _detect(path, pad_ratio)
    if points is None:
        return None
    return points, label, dataset_id, path


def collect_jobs(sources, classes, pad_ratio):
    class_index = {name: i for i, name in enumerate(classes)}
    jobs = []
    skipped = Counter()

    for dataset_id, (name, root) in enumerate(sources):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            sys.exit(f"Error: no such directory '{root}' (source '{name}')")

        # Some Kaggle archives nest the class folders one level deeper.
        entries = [e for e in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, e))]
        if len(entries) == 1:
            nested = os.path.join(root, entries[0])
            if any(os.path.isdir(os.path.join(nested, e)) for e in os.listdir(nested)):
                root = nested
                entries = [e for e in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, e))]

        for entry in entries:
            label_name = entry.strip().lower()
            if label_name not in class_index:
                skipped[label_name] += 1
                continue
            class_dir = os.path.join(root, entry)
            files = [
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ]
            for path in sorted(files, key=_natural_key):
                jobs.append((path, class_index[label_name], dataset_id, pad_ratio))

    if skipped:
        print(f"Skipped classes (outside the {len(classes)} requested): {sorted(skipped)}")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", action="append", required=True, metavar="NAME=PATH",
        help="Root of one dataset, with a directory per class. Repeatable.",
    )
    parser.add_argument("--out", default="landmarks.npz")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--pad", type=float, default=0.25,
                        help="Black border added before detection; these datasets crop tight")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--limit-per-class", type=int, default=0, help="0 = no limit")
    args = parser.parse_args()

    classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()]

    sources = []
    for spec in args.source:
        if "=" not in spec:
            sys.exit(f"Error: --source expects NAME=PATH, got '{spec}'")
        name, path = spec.split("=", 1)
        sources.append((name.strip(), path.strip()))

    jobs = collect_jobs(sources, classes, args.pad)

    if args.limit_per_class:
        kept, seen = [], Counter()
        for job in jobs:
            key = (job[2], job[1])
            if seen[key] < args.limit_per_class:
                kept.append(job)
                seen[key] += 1
        jobs = kept

    if not jobs:
        sys.exit("Error: found no images at all. Check the --source paths.")

    print(f"Images to process: {len(jobs):,}  |  classes: {len(classes)}  |  workers: {args.workers}")
    print("Running MediaPipe (once; after this, training never touches the disk)...")

    results = []
    attempted = Counter()
    for job in jobs:
        attempted[(job[2], job[1])] += 1

    if args.workers > 1:
        import multiprocessing as mp

        with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.pad, args.min_confidence)) as pool:
            for i, out in enumerate(pool.imap(_process, jobs, chunksize=32), 1):
                if out is not None:
                    results.append(out)
                if i % 2000 == 0 or i == len(jobs):
                    print(f"  {i:,}/{len(jobs):,}  detected {len(results):,}", flush=True)
    else:
        _init_worker(args.pad, args.min_confidence)
        for i, job in enumerate(jobs, 1):
            out = _process(job)
            if out is not None:
                results.append(out)
            if i % 2000 == 0 or i == len(jobs):
                print(f"  {i:,}/{len(jobs):,}  detected {len(results):,}", flush=True)

    if not results:
        sys.exit("Error: MediaPipe found no hands at all. Check that --source points at the "
                 "original photographs, not at pre-rendered skeletons.")

    points = np.stack([r[0] for r in results]).astype(np.float32)
    labels = np.array([r[1] for r in results], dtype=np.int16)
    datasets = np.array([r[2] for r in results], dtype=np.int8)
    paths = np.array([r[3] for r in results])

    np.savez_compressed(
        args.out,
        points=points,
        labels=labels,
        datasets=datasets,
        paths=paths,
        class_names=np.array(classes),
        dataset_names=np.array([s[0] for s in sources]),
    )

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nSaved {args.out}  ({len(points):,} samples, {size_mb:.1f} MB)")

    print("\nDetection rate by class and dataset:")
    detected = Counter((int(d), int(l)) for _, l, d, _ in results)
    for dataset_id, (name, _) in enumerate(sources):
        rows = [(classes[l], detected[(dataset_id, l)], attempted[(dataset_id, l)])
                for l in range(len(classes)) if attempted[(dataset_id, l)]]
        if not rows:
            continue
        total_ok = sum(r[1] for r in rows)
        total_all = sum(r[2] for r in rows)
        print(f"\n  [{name}]  {total_ok:,}/{total_all:,} = {total_ok / total_all:.1%}")
        weak = [(c, ok, all_) for c, ok, all_ in rows if all_ and ok / all_ < 0.6]
        for cls, ok, all_ in rows:
            print(f"    {cls}: {ok:5d}/{all_:5d}  {ok / all_:6.1%}")
        if weak:
            print(f"    WARNING: below 60% detection: {[w[0] for w in weak]}")
            print("    Those classes would reach training with very few samples.")


if __name__ == "__main__":
    main()
