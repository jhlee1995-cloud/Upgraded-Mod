"""
extract/download_isun.py -- fetch iSUN (energy-normal near-OOD) from HuggingFace.

iSUN = 8,925 natural scene images sampled from SUN. It is the energy-normal near-OOD
needed to test type-b detection (CLUSTER_DISTANCE fires, CONSENSUS silent) and the
last single-batch false-redundancy pair (CLUSTER_DISTANCE vs DEVIATION).

Saves images as flat PNG files into <volume>/datasets/isun/ so ImageFolderFlat reads them.

Usage:
  pip install datasets pillow
  python -m extract.download_isun --volume /workspace
  # then in extract: --isun
"""
import argparse
import os
import sys


def main(args):
    out_dir = os.path.join(args.volume, "datasets", "isun")
    os.makedirs(out_dir, exist_ok=True)

    # already populated?
    existing = [f for f in os.listdir(out_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(existing) >= args.min_images and not args.force:
        print(f"iSUN already present: {len(existing)} images in {out_dir} (use --force to redo)")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: need `pip install datasets pillow`")
        sys.exit(1)

    print("loading detectors/isun-ood from HuggingFace ...")
    # the dataset has a single split; images are in an 'image' column
    ds = load_dataset("detectors/isun-ood", split="train")
    print(f"  loaded {len(ds)} examples; columns = {ds.column_names}")

    # find the image column
    img_col = None
    for c in ds.column_names:
        if c.lower() in ("image", "img", "picture"):
            img_col = c
            break
    if img_col is None:
        # take the first column that yields a PIL image
        img_col = ds.column_names[0]
    print(f"  using image column: '{img_col}'")

    n = 0
    for i, ex in enumerate(ds):
        img = ex[img_col]
        # ex[img_col] is a PIL.Image when the feature is an Image()
        try:
            img = img.convert("RGB")
        except AttributeError:
            print(f"  example {i} is not a PIL image ({type(img)}); aborting")
            break
        img.save(os.path.join(out_dir, f"isun_{i:05d}.png"))
        n += 1
        if args.max_images and n >= args.max_images:
            break
        if n % 1000 == 0:
            print(f"  saved {n} ...")
    print(f"DONE: {n} iSUN images -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="/workspace")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--min-images", type=int, default=1000,
                    help="skip download if at least this many already present")
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
