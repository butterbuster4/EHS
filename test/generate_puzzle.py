import math
import cv2
import numpy as np
from utils import extract_tiles_from_image

# =========================
# 1. Configuration
# =========================
IMAGE_PATH = "./48577.jpg"  # Path to the source puzzle image
SAVE_PATH = "./test.jpg" # Path to save the scrambled result
GRID_SIZE = 3             # 3x3 grid
GAP = 0                   # Padding/Gap between tiles
TILE_SIZE = 96           # Dimension of each square tile

# =========================
# 2. Puzzle Reconstruction
# =========================
def form_puzzle_from_tiles(tiles, save_path="scrambled_puzzle.png"):
    """
    Combines a list of tiles into a single grid image and saves it.
    """
    if not tiles or len(tiles) == 0:
        raise ValueError("The input tiles list is empty!")

    # 1. Dynamically calculate grid size (e.g., 9 tiles -> 3x3, 16 tiles -> 4x4)
    num_tiles = len(tiles)
    grid_size = int(math.sqrt(num_tiles))
    
    if grid_size * grid_size != num_tiles:
        raise ValueError(f"Tile count ({num_tiles}) is not a perfect square; cannot form a grid!")

    # 2. Get dimensions of a single tile (Height, Width, Channels)
    h, w, c = tiles[0].shape

    # 3. Create a large canvas (solid black background)
    canvas = np.zeros((grid_size * h, grid_size * w, c), dtype=np.uint8)

    # 4. Paste tiles into the grid sequence
    for idx, tile in enumerate(tiles):
        r = idx // grid_size  # Current row index
        c_ = idx % grid_size  # Current column index
        
        # Paste the tile into the specific region on the canvas
        canvas[r*h:(r+1)*h, c_*w:(c_+1)*w, :] = tile

    # 5. Save the final image
    # Note: If tiles are in RGB format, OpenCV's imwrite requires BGR for correct colors
    if c == 3:
        save_canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    else:
        save_canvas = canvas

    cv2.imwrite(save_path, save_canvas)
    print(f"Grid puzzle generated and saved to: {save_path}")

    return canvas

# =========================
# 3. Execution Main
# =========================
if __name__ == "__main__":
    # Extract tile segments from the source image
    tiles = extract_tiles_from_image(
        IMAGE_PATH, 
        tile_per_side=GRID_SIZE, 
        tile_size=TILE_SIZE, 
        gap=GAP
    )
    
    # Randomly shuffle the tiles to create a scrambled state
    np.random.shuffle(tiles)
    
    # Form the puzzle canvas and save the image to disk
    form_puzzle_from_tiles(tiles, save_path=SAVE_PATH)