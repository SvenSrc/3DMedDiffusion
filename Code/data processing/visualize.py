import matplotlib.pyplot as plt

from preprocessing import (
    percentile_clip,
    crop_foreground,
    pad_to_cubic,
    resize_volume,
    normalize_to_range,
)


def plot_orthogonal_slices(volume, title="volume", cmap="gray"):
    d, h, w = (s // 2 for s in volume.shape)
    views = [("axial", volume[d]),
             ("coronal", volume[:, h]),
             ("sagittal", volume[:, :, w])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, img) in zip(axes, views):
        ax.imshow(img, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{title}  shape={volume.shape}  range=[{volume.min():.2f}, {volume.max():.2f}]")
    plt.tight_layout()
    plt.show()


def plot_preprocessing_steps(volume, target_size):
    # walk through the pipeline and show the center axial slice at each stage
    steps = [("original", volume)]

    v = percentile_clip(volume)
    steps.append(("clipped", v))
    v = crop_foreground(v)
    steps.append(("cropped", v))
    v = pad_to_cubic(v)
    steps.append(("padded", v))
    v = resize_volume(v, target_size)
    steps.append(("resized", v))
    v = normalize_to_range(v)
    steps.append(("normalized", v))

    fig, axes = plt.subplots(1, len(steps), figsize=(3 * len(steps), 4))
    for ax, (name, vol) in zip(axes, steps):
        center = vol.shape[0] // 2
        ax.imshow(vol[center], cmap="gray")
        ax.set_title(f"{name}\n{vol.shape}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def print_volume_stats(volume, name="volume"):
    print(f"{name}: shape={volume.shape}, dtype={volume.dtype}, "
          f"min={volume.min():.3f}, max={volume.max():.3f}, "
          f"mean={volume.mean():.3f}, std={volume.std():.3f}")
