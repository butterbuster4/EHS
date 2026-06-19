import os
import json
import random
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# Extract tiles
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


# Build ground-truth image from correctly ordered tiles
def build_gt_image(group_tiles, canvas_cols=3, tile_per_side=3, background_color=(220, 220, 220)):
    """
    Build ground-truth image from correctly ordered tiles.
    
    group_tiles:
        {
            group_id: [
                {"img": tile_img, "grid_pos": (row, col)},
                ...
            ]
        }

    Return:
        gt_canvas: All reconstructed puzzles arranged in rows.
    """

    group_ids = sorted(group_tiles.keys())

    # Tile size after erosion, e.g. 96 x 96 when tile_size=104 and gap=4
    sample_tile = group_tiles[group_ids[0]][0]["img"]
    tile_h, tile_w = sample_tile.shape[:2]

    puzzle_h = tile_per_side * tile_h
    puzzle_w = tile_per_side * tile_w

    num_groups = len(group_ids)

    # Arrange GT puzzles in a compact grid
    gt_cols = min(canvas_cols, num_groups)
    gt_rows = (num_groups + gt_cols - 1) // gt_cols

    gt_canvas = np.ones(
        (gt_rows * puzzle_h, gt_cols * puzzle_w, 3),
        dtype=np.uint8
    ) * np.array(background_color, dtype=np.uint8)

    for idx, gid in enumerate(group_ids):
        puzzle_canvas = np.ones(
            (puzzle_h, puzzle_w, 3),
            dtype=np.uint8
        ) * np.array(background_color, dtype=np.uint8)

        for t in group_tiles[gid]:
            row, col = t["grid_pos"]

            y1 = row * tile_h
            y2 = (row + 1) * tile_h
            x1 = col * tile_w
            x2 = (col + 1) * tile_w

            puzzle_canvas[y1:y2, x1:x2] = t["img"]

        out_r = idx // gt_cols
        out_c = idx % gt_cols

        y1 = out_r * puzzle_h
        y2 = (out_r + 1) * puzzle_h
        x1 = out_c * puzzle_w
        x2 = (out_c + 1) * puzzle_w

        gt_canvas[y1:y2, x1:x2] = puzzle_canvas

    return gt_canvas


# Make a single instance
def make_instance(instance_id, image_paths, out_dir,
                  tile_per_side=3, tile_size=104, gap=4,
                  canvas_cols=15, background_color=(220, 220, 220)):

    image_paths = list(image_paths)
    random.shuffle(image_paths)

    tiles_all = []
    group_tiles = {}

    # Read and crop images
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
        img_resized = cv2.resize(
            img_cropped,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA
        )

        tiles = extract_tiles(
            img_resized,
            tile_per_side=tile_per_side,
            tile_size=tile_size,
            gap=gap
        )

        group_tiles[gid] = []

        for t in tiles:
            tile_record = {
                "group_id": gid,
                "grid_pos": t["grid_pos"],
                "img": t["img"]
            }

            tiles_all.append(tile_record)

            group_tiles[gid].append({
                "grid_pos": t["grid_pos"],
                "img": t["img"]
            })

    # Build mixed input image
    random.shuffle(tiles_all)

    tile_h, tile_w, _ = tiles_all[0]["img"].shape
    N = len(tiles_all)

    rows = (N + canvas_cols - 1) // canvas_cols

    canvas = np.ones(
        (rows * tile_h, canvas_cols * tile_w, 3),
        dtype=np.uint8
    ) * np.array(background_color, dtype=np.uint8)

    meta = {
        "instance_id": instance_id,
        "num_groups": len(image_paths),
        "tile_per_side": tile_per_side,
        "tile_size": tile_size,
        "gap": gap,
        "eroded_tile_size": [tile_h, tile_w],
        "pieces": []
    }

    # Place mixed tiles
    for idx, t in enumerate(tiles_all):
        r = idx // canvas_cols
        c = idx % canvas_cols

        y1 = r * tile_h
        y2 = (r + 1) * tile_h
        x1 = c * tile_w
        x2 = (c + 1) * tile_w

        canvas[y1:y2, x1:x2] = t["img"]

        meta["pieces"].append({
            "piece_id": idx,
            "group_id": t["group_id"],
            "grid_pos": list(t["grid_pos"]),
            "mix_pos": [r, c],
            "bbox": [x1, y1, x2, y2]
        })

    # Build ground-truth image
    gt_canvas = build_gt_image(
        group_tiles=group_tiles,
        canvas_cols=canvas_cols // tile_per_side,
        tile_per_side=tile_per_side,
        background_color=background_color
    )

    # Save files
    inst_dir = Path(out_dir) / f"instance_{instance_id:06d}"
    inst_dir.mkdir(exist_ok=True, parents=True)

    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(inst_dir / "mix.png")
    Image.fromarray(cv2.cvtColor(gt_canvas, cv2.COLOR_BGR2RGB)).save(inst_dir / "gt.png")

    with open(inst_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return instance_id


# Parallel dataset builder for multiple mix configurations
def build_dataset_multi_mix(source_folder, out_folder, n_instances_per_mix=1000,
                            mix_list=[6, 12, 20, 30],
                            tile_per_side=3, tile_size=104, gap=4,
                            num_workers=8, canvas_cols=15, background_color=(220, 220, 220)):

    imgs = [
        str(Path(source_folder) / f)
        for f in os.listdir(source_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    assert len(imgs) >= max(mix_list), "Not enough images in source_folder."

    os.makedirs(out_folder, exist_ok=True)

    inst_counter = 0

    for mix_num in mix_list:
        print(f"Generating {n_instances_per_mix} instances for {mix_num}-mix...")

        tasks = []

        for _ in range(n_instances_per_mix):
            sel = random.sample(imgs, mix_num)
            tasks.append((inst_counter, sel))
            inst_counter += 1

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    make_instance,
                    inst_id,
                    sel,
                    out_folder,
                    tile_per_side,
                    tile_size,
                    gap,
                    canvas_cols,
                )
                for inst_id, sel in tasks
            ]

            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass

    print("Multi-mix dataset generation complete:", out_folder)


# Main entry point
if __name__ == "__main__":
    source_folder = "E:/MET_dataset/train"
    out_folder = "./"

    build_dataset_multi_mix(
        source_folder=source_folder,
        out_folder=out_folder,
        n_instances_per_mix=1,
        mix_list=[30],
        tile_per_side=3,
        tile_size=104,
        gap=4,
        num_workers=8,
        canvas_cols=9,
    )