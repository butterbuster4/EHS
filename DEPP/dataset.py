import os
import json
import random
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------
# Extract Tiles
# -----------------------------------------
def extract_tiles(image, tile_per_side=3, tile_size=104, gap=4):
    h, w = image.shape[:2]
    assert h >= tile_per_side * tile_size and w >= tile_per_side * tile_size
    tiles = []

    for row in range(tile_per_side):
        for col in range(tile_per_side):
            y1 = row * tile_size + gap
            y2 = (row + 1) * tile_size - gap
            x1 = col * tile_size + gap
            x2 = (col + 1) * tile_size - gap
            tile = image[y1:y2, x1:x2]
            tiles.append({
                "img": tile,
                "grid_pos": (row, col)
            })
    return tiles

# -----------------------------------------
# Make a single instance
# -----------------------------------------
def make_instance(instance_id, image_paths, out_dir,
                  tile_per_side=3, tile_size=104, gap=4,
                  canvas_cols=15, background_color=(220,220,220)):

    random.shuffle(image_paths)
    tiles_all = []

    # --- read & crop ---
    for gid, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            return None

        h, w = img.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        img_cropped = img[start_y:start_y + min_dim, start_x:start_x + min_dim]

        target_size = tile_per_side * tile_size
        img_resized = cv2.resize(img_cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

        tiles = extract_tiles(img_resized, tile_per_side=tile_per_side, tile_size=tile_size, gap=gap)
        for t in tiles:
            tiles_all.append({
                "group_id": gid,
                "grid_pos": t["grid_pos"],
                "img": t["img"]
            })

    random.shuffle(tiles_all)
    tile_h, tile_w, _ = tiles_all[0]["img"].shape
    N = len(tiles_all)
    rows = (N + canvas_cols - 1) // canvas_cols

    canvas = np.ones((rows * tile_h, canvas_cols * tile_w, 3), dtype=np.uint8) * 220

    meta = {
        "instance_id": instance_id,
        "pieces": []
    }

    # place tiles
    for idx, t in enumerate(tiles_all):
        r = idx // canvas_cols
        c = idx % canvas_cols
        y1, y2 = r * tile_h, (r + 1) * tile_h
        x1, x2 = c * tile_w, (c + 1) * tile_w
        canvas[y1:y2, x1:x2] = t["img"]

        meta["pieces"].append({
            "group_id": t["group_id"],
            "grid_pos": t["grid_pos"],
            "mix_pos": [r, c],
            "bbox": [x1, y1, x2, y2]
        })

    inst_dir = Path(out_dir) / f"instance_{instance_id:06d}"
    inst_dir.mkdir(exist_ok=True, parents=True)

    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(inst_dir / "mix.png")
    json.dump(meta, open(inst_dir / "meta.json", "w"), indent=2)

    return instance_id

# -----------------------------------------
# Parallel Dataset Builder for multiple mix
# -----------------------------------------
def build_dataset_multi_mix(source_folder, out_folder, n_instances_per_mix=1000,
                            mix_list=[6,12,20,30],
                            tile_per_side=3, tile_size=104, gap=4,
                            num_workers=8):

    imgs = [str(Path(source_folder) / f) for f in os.listdir(source_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    assert len(imgs) >= max(mix_list), "Not enough images in source_folder."

    os.makedirs(out_folder, exist_ok=True)
    inst_counter = 0

    for mix_num in mix_list:
        print(f"Generating {n_instances_per_mix} instances for {mix_num}-mix...")
        tasks = []
        for i in range(n_instances_per_mix):
            sel = random.sample(imgs, mix_num)
            tasks.append((inst_counter, sel))
            inst_counter += 1

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(make_instance, inst_id, sel, out_folder,
                                       tile_per_side, tile_size, gap) for inst_id, sel in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass

    print("Multi-mix dataset generation complete:", out_folder)

# -----------------------------------------
# Main
# -----------------------------------------
if __name__ == "__main__":
    source_folder = "./MET_dataset/train"
    out_folder = "./"
    build_dataset_multi_mix(
        source_folder=source_folder,
        out_folder=out_folder,
        n_instances_per_mix=1,  
        mix_list=[20],
        tile_per_side=3,
        tile_size=104,
        gap=4,
        num_workers=8
    )
    
    """out_folder = "./dataset/12_mix/train"
    build_dataset_multi_mix(
        source_folder=source_folder,
        out_folder=out_folder,
        n_instances_per_mix=9000,  
        mix_list=[12],
        tile_per_side=3,
        tile_size=104,
        gap=4,
        num_workers=8
    )
    
    out_folder = "./dataset/20_mix/train"
    build_dataset_multi_mix(
        source_folder=source_folder,
        out_folder=out_folder,
        n_instances_per_mix=9000,  
        mix_list=[20],
        tile_per_side=3,
        tile_size=104,
        gap=4,
        num_workers=8
    )
    
    out_folder = "./dataset/30_mix/train"
    build_dataset_multi_mix(
        source_folder=source_folder,
        out_folder=out_folder,
        n_instances_per_mix=9000,  
        mix_list=[30],
        tile_per_side=3,
        tile_size=104,
        gap=4,
        num_workers=8
    )"""
