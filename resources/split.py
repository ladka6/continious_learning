"""Split imagenet-r/ (flat 200 class folders) into train/ and test/ (80/20).

Run once:
    python resources/split_imagenet_r.py
"""
import os
import random
import shutil
from pathlib import Path

SRC = Path("data/imagenet-r")
DST = Path("data/imagenet-r-split")
SEED = 1993
TEST_RATIO = 0.2


def main():
    assert SRC.exists(), f"{SRC} not found"
    random.seed(SEED)
    classes = sorted(d for d in os.listdir(SRC) if (SRC / d).is_dir())
    print(f"Found {len(classes)} classes")
    for cls in classes:
        imgs = sorted(os.listdir(SRC / cls))
        random.shuffle(imgs)
        n_test = int(len(imgs) * TEST_RATIO)
        test_imgs, train_imgs = imgs[:n_test], imgs[n_test:]
        for split, names in [("train", train_imgs), ("test", test_imgs)]:
            out = DST / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for n in names:
                shutil.copy(SRC / cls / n, out / n)
    print(f"Done. Output: {DST}")


if __name__ == "__main__":
    main()