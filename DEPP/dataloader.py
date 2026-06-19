import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def process_instance_folder(inst_folder):
    """Load one instance folder and return:
    - imgs: np.array (num_tiles, H, W, 3) uint8
    - labels: np.array (num_tiles,) group_ids
    """
    inst_folder = Path(inst_folder)
    meta_path = inst_folder / "meta.json"
    if not meta_path.exists():
        return None

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)

        mix_img = np.array(Image.open(inst_folder / "mix.png"))

        tiles_imgs = []
        labels = []
        for p in meta["pieces"]:
            x1, y1, x2, y2 = p["bbox"]
            tile = mix_img[y1:y2, x1:x2]
            tiles_imgs.append(tile)
            labels.append(p["group_id"])

        tiles_imgs = np.stack(tiles_imgs).astype(np.uint8)
        labels = np.array(labels, dtype=np.int32)
        return tiles_imgs, labels

    except Exception:
        return None


def save_pt_bundle(args):
    imgs, labels, out_path = args
    imgs_tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    labels_tensor = torch.from_numpy(labels).long()

    torch.save(
        {"images": imgs_tensor, "labels": labels_tensor},
        out_path,
        _use_new_zipfile_serialization=True
    )


def convert_dataset_to_pt(dataset_folder, out_folder, max_instances=None, start_index=0, num_workers=8):
    """Convert a dataset folder (train/test) to .pt files.
    max_instances: select only the first N instance folders.
    """
    dataset_folder = Path(dataset_folder)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # load all instance folders
    instance_folders = sorted([f for f in dataset_folder.iterdir() if f.is_dir()])

    # if max_instances is specified, limit number
    if max_instances is not None:
        instance_folders = instance_folders[:max_instances]

    print(f"Processing {len(instance_folders)} instances with {num_workers} workers...")

    # ---- STEP 1: parallel load all instances ----
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_instance_folder, instance_folders),
                            total=len(instance_folders),
                            desc="Loading"))

    # ---- STEP 2: prepare saving tasks ----
    save_tasks = []
    idx = start_index
    for res in results:
        if res is None:
            continue
        imgs, labels = res
        out_path = out_folder / f"instance_{idx:06d}.pt"
        save_tasks.append((imgs, labels, out_path))
        idx += 1

    # ---- STEP 3: parallel save .pt files ----
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(save_pt_bundle, save_tasks),
                  total=len(save_tasks),
                  desc="Saving"))

    print(f"Saved {len(save_tasks)} instances to {out_folder}")


if __name__ == "__main__":
    convert_dataset_to_pt(
        "./dataset/20_mix/train",
        "./clustering_dataset",
        max_instances=2000,
        start_index=13000,
        num_workers=8,
    )

    convert_dataset_to_pt(
        "./dataset/30_mix/train",
        "./clustering_dataset",
        max_instances=2000,
        start_index=15000,
        num_workers=8,
    )
