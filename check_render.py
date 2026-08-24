"""Contact sheet of what the network actually receives.

This is the check that matters most in this project, and it is worth running before
committing to a training run. The whole design rests on training and inference drawing
the same picture; if the sheet this produces does not look like the "AI View (Skeleton)"
window in the live translator, nothing downstream is meaningful.

What to look for:
  * every hand centred, at the same size, filling most of its tile;
  * finger colours consistent tile to tile (thumb cream, index purple, middle yellow,
    ring green, pinky blue, palm joints red);
  * no hand clipped at a tile edge;
  * with --augment, the same properties still hold after perturbation.

    python check_render.py                      # one sample per class
    python check_render.py --augment            # after augmentation
    python check_render.py --class a --count 24 # 24 variants of one letter
"""

import argparse
import os

import cv2
import numpy as np

import preprocessing as pp


def contact_sheet(images, labels, tile, columns):
    rows = (len(images) + columns - 1) // columns
    label_height = 16
    cell = tile + label_height
    sheet = np.zeros((rows * cell, columns * tile, 3), dtype=np.uint8)

    for i, (image, label) in enumerate(zip(images, labels)):
        r, c = divmod(i, columns)
        y, x = r * cell, c * tile
        sheet[y:y + tile, x:x + tile] = cv2.resize(image, (tile, tile),
                                                   interpolation=cv2.INTER_NEAREST)
        cv2.putText(sheet, str(label), (x + 4, y + tile + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--landmarks", default="landmarks.npz")
    parser.add_argument("--out", default="docs/render-check.png")
    parser.add_argument("--size", type=int, default=pp.MODEL_INPUT_SIZE)
    parser.add_argument("--tile", type=int, default=96, help="On-screen size of each tile")
    parser.add_argument("--columns", type=int, default=13)
    parser.add_argument("--augment", action="store_true", help="Render through augment_landmarks")
    parser.add_argument("--class", dest="only", default="", help="Show many samples of one class")
    parser.add_argument("--count", type=int, default=0, help="With --class: how many samples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.landmarks):
        raise SystemExit(f"Error: '{args.landmarks}' not found. Run prepare_dataset.py first.")

    data = np.load(args.landmarks, allow_pickle=False)
    points = data["points"].astype(np.float32)
    labels = data["labels"].astype(int)
    class_names = [str(c) for c in data["class_names"]]

    rng = np.random.default_rng(args.seed)

    if args.only:
        if args.only not in class_names:
            raise SystemExit(f"Error: unknown class '{args.only}'. Available: {class_names}")
        pool = np.where(labels == class_names.index(args.only))[0]
        if len(pool) == 0:
            raise SystemExit(f"Error: no samples for class '{args.only}'.")
        count = args.count or min(26, len(pool))
        chosen = rng.choice(pool, size=min(count, len(pool)), replace=False)
    else:
        # One representative per class, so the sheet doubles as a look at the alphabet.
        chosen = []
        for label in range(len(class_names)):
            pool = np.where(labels == label)[0]
            if len(pool):
                chosen.append(rng.choice(pool))
        chosen = np.array(chosen)

    images, tile_labels = [], []
    for index in chosen:
        pts = points[index]
        if args.augment:
            pts = pp.augment_landmarks(pts, rng)
        images.append(pp.render_skeleton(pts, size=args.size))
        tile_labels.append(class_names[labels[index]])

    sheet = contact_sheet(images, tile_labels, args.tile, args.columns)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # render_skeleton returns RGB; cv2.imwrite expects BGR.
    cv2.imwrite(args.out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    filled = (sheet.sum(axis=2) > 30).mean()
    print(f"{len(images)} samples{' (augmented)' if args.augment else ''} -> {args.out}")
    print(f"model input {args.size}x{args.size}, canvas {pp.CANVAS_SIZE}, hand fill {pp.HAND_FILL:.0%}")
    print(f"non-black pixels: {filled:.1%}")
    print("\nCompare this against the 'AI View (Skeleton)' window in realtime_translator.py.")
    print("They must look the same. If they do not, the model is being trained on one")
    print("kind of picture and asked to classify another.")


if __name__ == "__main__":
    main()
