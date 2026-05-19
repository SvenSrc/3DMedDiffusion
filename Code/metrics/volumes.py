import json
from pathlib import Path

import nibabel as nib
import numpy as np


def load_volume(path):
    return nib.load(str(path)).get_fdata().astype(np.float32)


def list_volumes(directory):
    files = sorted(Path(directory).glob("*.nii.gz"))
    assert len(files) > 0, f"no .nii.gz files in {directory}"
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
