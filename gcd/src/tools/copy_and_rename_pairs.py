import argparse
import os
import re
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Copy direction_transfer outputs to a new folder with standardized names."
    )
    parser.add_argument("--src-dir", required=True, help="Path to direction_transfer subdir (contains original_*.png, inpaint_*.png)")
    parser.add_argument("--dst-dir", required=True, help="Destination directory to write renamed copies")
    parser.add_argument("--only-pairs", action="store_true", help="Copy only when both original_k.png and inpaint_k.png exist")
    args = parser.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Patterns: original_1.png, inpaint_1.png
    pat_orig = re.compile(r"^original_(\d+)\.png$")
    pat_inpt = re.compile(r"^inpaint_(\d+)\.png$")

    indices_orig = {}
    indices_inpt = {}

    for entry in os.listdir(src):
        m1 = pat_orig.match(entry)
        if m1:
            idx = int(m1.group(1))
            indices_orig[idx] = src / entry
            continue
        m2 = pat_inpt.match(entry)
        if m2:
            idx = int(m2.group(1))
            indices_inpt[idx] = src / entry

    # Decide which indices to process
    if args.only_pairs:
        all_idx = sorted(set(indices_orig.keys()).intersection(indices_inpt.keys()))
    else:
        all_idx = sorted(set(indices_orig.keys()).union(indices_inpt.keys()))

    copied_images = 0
    copied_inpaints = 0

    for i in all_idx:
        z = str(i).zfill(5)
        if i in indices_orig:
            src_path = indices_orig[i]
            tgt_path = dst / f"images_{z}.png"
            shutil.copy2(src_path, tgt_path)
            copied_images += 1
        if i in indices_inpt:
            src_path = indices_inpt[i]
            tgt_path = dst / f"inpaints_{z}.png"
            shutil.copy2(src_path, tgt_path)
            copied_inpaints += 1

    print(f"Done. Wrote {copied_images} images_*.png and {copied_inpaints} inpaints_*.png into {dst}")


if __name__ == "__main__":
    main()


