import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

def find_models(runs_dir: Path):
    models = []
    for root, _, files in os.walk(runs_dir):
        for f in files:
            if f == "best.pt":
                models.append(Path(root) / f)
    return models


def run_inference_on_model(model_path: Path, images: list, batch_size: int, out_root: Path):
    print(f"-> Loading model: {model_path}")
    model = YOLO(str(model_path))
    # build a cleaner model identifier:
    # if the model lives in a 'weights' subfolder (common with ultralytics runs),
    # use the parent directory name + '_weights' to avoid embedding the entire path
    if model_path.parent.name == "weights" and model_path.parent.parent.name:
        model_name = f"{model_path.parent.parent.name}_weights"
    else:
        model_name = model_path.parent.name or model_path.stem
    model_out = out_root / f"{model_name}"
    model_out.mkdir(parents=True, exist_ok=True)

    total = len(images)
    for i in range(0, total, batch_size):
        batch = images[i : i + batch_size]
        print(f"=Running batch {i//batch_size + 1} ({len(batch)} images)...")
        try:
            results = model(batch)  # list of Results
        except Exception as e:
            print(f"   ERROR running model {model_path} on batch starting at {i}: {e}", file=sys.stderr)
            continue

        for idx, result in enumerate(results):
            src_img = Path(batch[idx])
            out_name = f"{src_img.stem}__{model_name}{src_img.suffix}"
            out_path = model_out / out_name
            try:
                # get annotated image ndarray from ultralytics and save via PIL to avoid default overwrite
                img_arr = result.plot()  # annotated image as numpy array (RGB)
                Image.fromarray(img_arr).save(out_path)
                print(f"      Saved: {out_path}")
            except Exception as e:
                # Do not call result.save() (can create a second file with different naming).
                print(f"      Failed to save annotated image for {src_img}: {e}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Run inference with every best.pt under a runs directory.")
    p.add_argument("--runs-dir", type=str, default="runs", help="Root runs directory to search for best.pt")
    p.add_argument("--images", nargs="*", default=[], help="One or more image files or directories")
    p.add_argument("--images-dir", type=str, help="Directory to recursively collect images from")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for model(...) calls")
    p.add_argument("--out", type=str, default="inference/results", help="Output root directory")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_root = Path(args.out)
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images_dir = Path(args.images_dir) if args.images_dir else Path("inference")
    images = []
    # collect from images_dir (defaults to ./inference)
    if images_dir.is_dir():
        for f in sorted(images_dir.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                images.append(str(f))
    # include any explicit --images paths (files or directories)
    if args.images:
        for p in args.images:
            p = Path(p)
            if p.is_dir():
                for f in sorted(p.iterdir()):
                    if f.suffix.lower() in IMG_EXTS:
                        images.append(str(f))
            elif p.is_file():
                images.append(str(p))
    # dedupe while preserving order
    seen = set()
    images = [x for x in images if not (x in seen or seen.add(x))]

    if not images:
        print("No input images found. Provide --images or --images-dir.", file=sys.stderr)
        sys.exit(1)

    models = find_models(runs_dir)
    if not models:
        print(f"No best.pt models found under {runs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(models)} model(s). Running inference on {len(images)} image(s).")
    for m in models:
        try:
            run_inference_on_model(m, images, args.batch_size, out_root)
        except Exception as e:
            print(f"Error processing model {m}: {e}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()
