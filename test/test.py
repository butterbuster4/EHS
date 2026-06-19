import argparse
import cv2
import numpy as np


def cut_image_into_tiles_and_rearrange(
    image_path,
    output_shape=None,
    order=None,
    tile_size=96,
    save_path=None,
):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    if h % tile_size != 0 or w % tile_size != 0:
        raise ValueError("Image dimensions must be multiples of tile_size.")

    num_rows = h // tile_size
    num_cols = w // tile_size
    total_tiles = num_rows * num_cols

    if order is None:
        flat_order = list(range(total_tiles))
    else:
        order_arr = np.array(order, dtype=np.int32)
        if order_arr.ndim == 1:
            flat_order = order_arr.tolist()
        elif order_arr.ndim == 2:
            flat_order = order_arr.flatten().tolist()
        else:
            raise ValueError("Order must be a 1D or 2D list of tile indices.")

    if len(flat_order) != total_tiles:
        raise ValueError(
            f"Order length must equal number of tiles ({total_tiles}), got {len(flat_order)}."
        )

    if output_shape is None:
        output_rows = num_rows
        output_cols = num_cols
    else:
        output_rows, output_cols = output_shape
        if output_rows * output_cols != total_tiles:
            raise ValueError(
                f"Output shape {output_shape} does not match tile count {total_tiles}."
            )

    tiles = []
    for row in range(num_rows):
        for col in range(num_cols):
            y1 = row * tile_size
            y2 = y1 + tile_size
            x1 = col * tile_size
            x2 = x1 + tile_size
            tiles.append(image[y1:y2, x1:x2].copy())

    arranged = np.zeros((output_rows * tile_size, output_cols * tile_size, 3), dtype=image.dtype)
    for new_pos, tile_idx in enumerate(flat_order):
        if tile_idx < 0 or tile_idx >= total_tiles:
            raise ValueError(f"Order contains invalid tile index: {tile_idx}")
        row = new_pos // output_cols
        col = new_pos % output_cols
        arranged[row*tile_size:(row+1)*tile_size,
                 col*tile_size:(col+1)*tile_size] = tiles[tile_idx]

    if save_path is not None:
        arranged_bgr = cv2.cvtColor(arranged, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, arranged_bgr)
    return tiles, arranged


if __name__ == "__main__":
    _, arranged = cut_image_into_tiles_and_rearrange(
        "./mix.png",
        output_shape=(15, 18),
        order=None,
        tile_size=96,
        save_path="./arranged.png",
    )

    