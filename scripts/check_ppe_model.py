"""Check a PPE weights file (Track C) before using it.

    python scripts/check_ppe_model.py models/custom_ppe.pt [image.jpg]

Prints the model's class names, how each maps to this project's shared vocabulary, and (optionally) what it
detects in one image. Classes marked "ignored" are loaded but never used by the pipeline.

SECURITY: .pt files are Python pickles and can run code when loaded. Only use weights from a source you trust,
and scan unknown files first (for example with `picklescan`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from siteeye.vision import canonical_label  # noqa: E402

USED = {"person", "hard hat", "no hard hat", "safety vest", "no safety vest", "safety hook", "harness lanyard",
        "excavator", "forklift", "crane hook", "heavy machinery", "ladder", "scaffolding"}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    weights = Path(sys.argv[1])
    if not weights.is_file():
        print(f"not found: {weights}")
        return 1
    from ultralytics import YOLO

    model = YOLO(str(weights))
    print(f"{weights.name}: {len(model.names)} classes, task={model.task}\n")
    print(f"{'id':>3}  {'model class':<24} -> {'pipeline label':<18} status")
    useful = 0
    for i, name in model.names.items():
        label = canonical_label(name)
        ok = label in USED
        useful += ok
        print(f"{i:>3}  {name:<24} -> {label:<18} {'used' if ok else 'ignored'}")
    print(f"\n{useful}/{len(model.names)} classes are used.")
    if not any(canonical_label(n) in {"hard hat", "no hard hat"} for n in model.names.values()):
        print("WARNING: no helmet class found; Track C will not improve helmet detection.")
    if len(sys.argv) > 2:
        r = model.predict(sys.argv[2], conf=0.25, verbose=False)[0]
        print("\nDetections in", sys.argv[2])
        for b in r.boxes:
            n = r.names[int(b.cls)]
            print(f"  {n:<24} {float(b.conf):.2f}  -> {canonical_label(n)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
