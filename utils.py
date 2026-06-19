import json
import math
from anyio import Path
import cv2
import numpy as np

def extract_tiles_from_image(image_path, tile_per_side=3, tile_size=104, gap=4):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    img_cropped = image[start_y:start_y + min_dim, start_x:start_x + min_dim]
    target_size = tile_per_side * tile_size
    img_resized = cv2.resize(img_cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    tiles = []
    
    for row in range(tile_per_side):
        for col in range(tile_per_side):
            y1 = row * tile_size + gap
            y2 = (row + 1) * tile_size - gap
            x1 = col * tile_size + gap
            x2 = (col + 1) * tile_size - gap
            tile = img_resized[y1:y2, x1:x2].copy()
            tiles.append(tile)
    return tiles

def split_puzzle_into_tiles(puzzle, tile_size=96):
    puzzle = cv2.imread(puzzle)
    puzzle = cv2.cvtColor(puzzle, cv2.COLOR_BGR2RGB)
    h, w = puzzle.shape[:2]
    
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = puzzle[y:y+tile_size, x:x+tile_size].copy()
            tiles.append(tile)
    return tiles

def split_tiles_according_to_labels(tiles, labels):
    clusters = {}
    for tile, label in zip(tiles, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(tile)
    return clusters

def show_tiles_in_grid(tiles, tile_size=96):
    num_tiles = len(tiles)
    
    cols = int(np.ceil(np.sqrt(num_tiles)))
    rows = int(np.ceil(num_tiles / cols))
    
    grid = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        y = r * tile_size
        x = c * tile_size
        grid[y:y+tile_size, x:x+tile_size] = tile

    cv2.imshow("All Tiles", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_clusters(clusters, tile_size=96):
    for label, tiles in clusters.items():
        show_tiles_in_grid(tiles, tile_size)

def tiles_to_grid(tiles, tile_size=96, max_cols=3):
    num_tiles = len(tiles)
    if num_tiles == 0:
        return None

    cols = min(max_cols, num_tiles)
    rows = int(np.ceil(num_tiles / cols))

    grid = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        y = r * tile_size
        x = c * tile_size

        tile_small = cv2.resize(tile, (tile_size, tile_size))
        grid[y:y+tile_size, x:x+tile_size] = tile_small

    return grid


def show_clusters_multi_grid(clusters, tile_size=35, max_cols=3, cluster_cols=6, label_height=22, save_path=None):
    cluster_ids = list(clusters.keys())
    num_clusters = len(cluster_ids)

    cluster_rows = math.ceil(num_clusters / cluster_cols)

    cluster_blocks = []
    i = 0
    
    for label in cluster_ids:
        tiles = clusters[label]

        grid = tiles_to_grid(tiles, tile_size=tile_size, max_cols=max_cols)
        if grid is None:
            continue

        h, w = grid.shape[:2]

        label_img = np.ones((label_height, w, 3), dtype=np.uint8) * 235
        cv2.putText(
            label_img,
            f"Cluster {i}",
            (5, int(label_height * 0.5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 1
        )
        i += 1
        block = np.vstack([label_img, grid])
        cluster_blocks.append(block)

    block_heights = [b.shape[0] for b in cluster_blocks]
    block_widths  = [b.shape[1] for b in cluster_blocks]

    max_h = max(block_heights)
    max_w = max(block_widths)

    canvas = np.zeros((cluster_rows * max_h, cluster_cols * max_w, 3), dtype=np.uint8)

    for idx, block in enumerate(cluster_blocks):
        r = idx // cluster_cols
        c = idx % cluster_cols

        h, w = block.shape[:2]

        y = r * max_h
        x = c * max_w

        canvas[y:y+h, x:x+w] = block

    # 保存图片到本地
    if save_path is not None:
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, canvas_bgr)
        print(f"Clustering result saved to: {save_path}")

    cv2.imshow("Clusters Grid View", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def load_meta(json_path):
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        meta = json.load(f)
    return meta

def get_gt_labels(meta):
    labels = [p["group_id"] for p in meta["pieces"]]
    return np.array(labels, dtype=np.int32)
