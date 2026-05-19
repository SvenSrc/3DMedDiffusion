import json
from pathlib import Path

import nibabel as nib
import numpy as np


def load_volume(path):
    return nib.load(str(path)).get_fdata().astype(np.float32)


def list_volumes(directory, pattern="*.nii.gz"):
    # pattern lets pipeline mode pick only e.g. "brain_*.nii.gz" out of a
    # folder that also contains "tumor_*.nii.gz".
    files = sorted(Path(directory).glob(pattern))
    assert len(files) > 0, f"no files matching {pattern!r} in {directory}"
    return files


def pair_random(real_files, gen_files, seed=42):
    # unconditional: every generated volume gets a random real partner.
    # with replacement only if there are fewer reals than generated samples.
    rng = np.random.default_rng(seed)
    replace = len(real_files) < len(gen_files)
    idx = rng.choice(len(real_files), size=len(gen_files), replace=replace)
    return [(real_files[i], gen_files[k]) for k, i in enumerate(idx)]


def pair_by_stats(real_dir, gen_files, stats_json):
    # seg2mri: stats.json from generate_conditional.ipynb records the seg-mask
    # stem used as conditioning for each generated sample. The real T1N volume
    # shares that stem, so we can recover the true ground-truth pair.
    real_dir = Path(real_dir)
    stats = json.loads(Path(stats_json).read_text())
    cond_by_filename = {s["filename"]: s["cond"] for s in stats["per_sample"]}

    pairs = []
    for gen in gen_files:
        stem = cond_by_filename[gen.name]
        real = real_dir / f"{stem}.nii.gz"
        if not real.exists():
            raise FileNotFoundError(f"no real volume for stem '{stem}' in {real_dir}")
        pairs.append((real, gen))
    return pairs


def pair_random_by_class(real_files, real_labels_csv, gen_files, gen_stats_json, seed=42):
    # novel_segmentation: each generated seg-map is conditioned on a size
    # class ("class0" / "class1" / "class2"). The real labels are stored in
    # a separate CSV with columns (filename, size_label). Pair every
    # generated map with a random real map of the same class so Dice
    # compares like with like.
    import pandas as pd

    label_map = pd.read_csv(real_labels_csv).set_index("filename")["size_label"].to_dict()
    real_by_class = {}
    for f in real_files:
        if f.name in label_map:
            real_by_class.setdefault(int(label_map[f.name]), []).append(f)

    stats = json.loads(Path(gen_stats_json).read_text())
    gen_class = {s["filename"]: int(s["cond"].replace("class", "")) for s in stats["per_sample"]}

    rng = np.random.default_rng(seed)
    pairs = []
    for gen in gen_files:
        cls = gen_class[gen.name]
        candidates = real_by_class.get(cls, [])
        if not candidates:
            raise ValueError(f"no real seg-maps with size class {cls} in {real_labels_csv}")
        real = candidates[rng.integers(len(candidates))]
        pairs.append((real, gen))
    return pairs
