import queue
import cv2
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from DEPP.DEPP import DEPP
from RLRS.RLRS import RLRS
from utils import load_meta, get_gt_labels

MODEL_PATH   = "./model/dscl_vit_epoch_40.pth"
TILE_SIZE    = 96
HORI_MODEL_PATH = "./model/hori_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth"
VRTI_MODEL_PATH = "./model/vrti_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth"
DQN_PATH     = "./model/dqn_jigsaw_final.zip"
PDN_PATH     = "./model/best_efficientnetb3.pth"

MAX_REFINE_STEPS    = 500
NUM_WORKERS         = 4
RL_MAX_STEPS        = 20
SUCCESS_THRESHOLD   = 0.9
EARLY_STOP_PATIENCE = 10


def accuracy(meta, original_indices_for_all_board):
    pieces = meta["pieces"]
    
    group_ids = set(p["group_id"] for p in pieces)
    gt_boards_by_puzzle = {}
    
    for puzzle_id in sorted(group_ids):
        grid_pos_to_original_index = {}
        for original_idx, piece in enumerate(pieces):
            if piece["group_id"] == puzzle_id:
                grid_pos = tuple(piece["grid_pos"])  # (row, col)
                grid_pos_to_original_index[grid_pos] = original_idx
        
        gt_board = []
        for row in range(3):
            for col in range(3):
                original_idx = grid_pos_to_original_index[(row, col)]
                gt_board.append(original_idx)
        
        gt_boards_by_puzzle[puzzle_id] = gt_board
    
    perfect_accuracy=0
    neighbor_accuracy=0
    absolute_accuracy=0
    
    # perfect accuracy
    perfect_count = 0
    for puzzle in original_indices_for_all_board:
        if puzzle in gt_boards_by_puzzle.values():
            perfect_count += 1
    perfect_accuracy = perfect_count / len(gt_boards_by_puzzle.values())
    
    # neighbor accuracy
    valid_right_edges = set()
    valid_down_edges = set()
    
    for gt_board in gt_boards_by_puzzle.values():
        for row in range(3):
            for col in range(3):
                pos = row * 3 + col
                current_idx = gt_board[pos]
                
                if col < 2: 
                    right_idx = gt_board[pos + 1]
                    valid_right_edges.add((current_idx, right_idx))
                
                if row < 2: 
                    down_idx = gt_board[pos + 3]
                    valid_down_edges.add((current_idx, down_idx))

    correct_edges = 0
    total_edges = 0
    
    for puzzle in original_indices_for_all_board:
        for row in range(3):
            for col in range(3):
                pos = row * 3 + col
                current_idx = puzzle[pos]
                
                if col < 2:
                    right_idx = puzzle[pos + 1]
                    if (current_idx, right_idx) in valid_right_edges:
                        correct_edges += 1
                    total_edges += 1
                
                if row < 2:
                    down_idx = puzzle[pos + 3]
                    if (current_idx, down_idx) in valid_down_edges:
                        correct_edges += 1
                    total_edges += 1
                
    neighbor_accuracy = correct_edges / total_edges if total_edges > 0 else 0
    
    # absolute accuracy
    index_to_group = {idx: p["group_id"] for idx, p in enumerate(pieces)}
    
    total_correct_tiles = 0
    total_tiles = 0
    
    for puzzle in original_indices_for_all_board:
        group_counts = {}
        for idx in puzzle:
            gid = index_to_group[idx]
            group_counts[gid] = group_counts.get(gid, 0) + 1
            
        major_group_id = max(group_counts, key=group_counts.get)
        
        major_gt_board = gt_boards_by_puzzle[major_group_id]
        
        for pos in range(9):
            predicted_idx = puzzle[pos]
            gt_idx = major_gt_board[pos]
            
            if predicted_idx == gt_idx:
                total_correct_tiles += 1
                
        total_tiles += 9
        
    absolute_accuracy = total_correct_tiles / total_tiles if total_tiles > 0 else 0
    
    
    return gt_boards_by_puzzle.values(), perfect_accuracy, neighbor_accuracy, absolute_accuracy
    
def get_island_chains(num_tiles, must_link):
    G = nx.Graph()
    G.add_nodes_from(range(num_tiles))
    G.add_edges_from(must_link)
    islands = [sorted(list(c)) for c in nx.connected_components(G)]
    islands.sort(key=len, reverse=True)
    return islands


def build_ground_truth_adjacency(meta):
    """Build adjacency sets from ground truth tile positions."""
    pieces = meta["pieces"]
    groups = {}
    for idx, piece in enumerate(pieces):
        group_id = piece["group_id"]
        grid_pos = tuple(piece["grid_pos"])
        groups.setdefault(group_id, {})[grid_pos] = idx

    adjacency = {idx: set() for idx in range(len(pieces))}
    for pos_to_idx in groups.values():
        for (row, col), idx in pos_to_idx.items():
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor_pos = (row + dr, col + dc)
                if neighbor_pos in pos_to_idx:
                    adjacency[idx].add(pos_to_idx[neighbor_pos])
    return adjacency


def annotate_incorrect_edges(image, best_board, feature_indices, meta, line_color=(255, 0, 0), thickness=6):
    """Annotate internal cluster edges that are incorrect in the assembled image."""
    if image is None or best_board is None or feature_indices is None:
        return image

    adjacency = build_ground_truth_adjacency(meta)
    num_pieces = len(meta["pieces"])
    grid_size = int(np.sqrt(len(best_board)))
    h, w = image.shape[:2]
    tile_h = h // grid_size
    tile_w = w // grid_size
    annotated = image.copy()

    for pos in range(len(best_board)):
        current_local = int(best_board[pos])
        if current_local < 0 or current_local >= len(feature_indices):
            continue
        current_global = feature_indices[current_local]
        if current_global < 0 or current_global >= num_pieces:
            continue
        r, c = divmod(pos, grid_size)

        if c < grid_size - 1:
            right_local = int(best_board[pos + 1])
            if right_local < 0 or right_local >= len(feature_indices):
                continue
            right_global = feature_indices[right_local]
            if right_global < 0 or right_global >= num_pieces:
                continue
            if right_global not in adjacency[current_global]:
                x = (c + 1) * tile_w
                y1 = r * tile_h
                y2 = (r + 1) * tile_h
                cv2.line(annotated, (x, y1), (x, y2), line_color, thickness)

        if r < grid_size - 1:
            down_local = int(best_board[pos + grid_size])
            if down_local < 0 or down_local >= len(feature_indices):
                continue
            down_global = feature_indices[down_local]
            if down_global < 0 or down_global >= num_pieces:
                continue
            if down_global not in adjacency[current_global]:
                y = (r + 1) * tile_h
                x1 = c * tile_w
                x2 = (c + 1) * tile_w
                cv2.line(annotated, (x1, y), (x2, y), line_color, thickness)

    return annotated


def save_image(images, path, cols=6):
    if not images:
        print(f"[INFO] No images to save for: {path}")
        return

    num_imgs = len(images)
    rows = (num_imgs + cols - 1) // cols
    h, w, c = images[0].shape

    canvas = np.zeros((rows * h, cols * w, c), dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // cols
        c_idx = idx % cols
        canvas[r*h:(r+1)*h, c_idx*w:(c_idx+1)*w] = img

    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Saved {num_imgs} images to: {path}")


def build_cluster_data_from_result(result):
    labels = result["labels"]
    tiles = result["tiles"]

    cluster_data = {}
    for idx, label in enumerate(labels):
        label = int(label)
        if label not in cluster_data:
            cluster_data[label] = {
                "tiles": [],
                "feature_indices": [],
            }
        cluster_data[label]["tiles"].append(tiles[idx])
        cluster_data[label]["feature_indices"].append(idx)

    return cluster_data

def make_rlrs_instance():
    return RLRS(
        model_hori_path=HORI_MODEL_PATH,
        model_vrti_path=VRTI_MODEL_PATH,
        dqn_path=DQN_PATH,
        pdn_path=PDN_PATH
    )


def solve_single_cluster(rlrs, cluster_id, cluster_info, old_cluster_id, max_steps=50):
    tiles = cluster_info["tiles"]
    feature_indices = cluster_info["feature_indices"]

    result_img, must_link, cannot_link, best_board, S_LR, S_UD = rlrs.solve(
        image_path=None,
        tiles=tiles,
        max_steps=max_steps
    )

    return {
        "cluster_id": cluster_id,
        "old_cluster_id": old_cluster_id,
        "result_img": result_img,
        "must_link": must_link,
        "cannot_link": cannot_link,
        "score": rlrs.best_global_score,
        "local_score": rlrs.best_local_score,
        "feature_indices": feature_indices,
        "tiles": tiles,
        "best_board": best_board,
    }

# =========================
# Main Puzzle Solving Logic
# =========================
def solve_puzzle(
    image_path,
    meta_path,
    output_prefix="./output/output",
    max_refine_steps=MAX_REFINE_STEPS,
    num_workers=NUM_WORKERS,
    rl_max_steps=RL_MAX_STEPS,
    success_threshold=SUCCESS_THRESHOLD,
    highlight_wrong_edges=False,
    save_annotated_cluster_images=False,
    early_stop_patience=EARLY_STOP_PATIENCE,
    depp=None,          
    rlrs_pool=None,   
):
    meta      = load_meta(meta_path)
    gt_labels = get_gt_labels(meta)
    num_total_puzzles = len(np.unique(gt_labels))

    original_indices_for_all_board = []

    # If DEPP or RLRS pool is not provided, create them
    _own_depp = depp is None
    _own_pool = rlrs_pool is None

    if _own_depp:
        depp = DEPP(checkpoint_path=MODEL_PATH, embedding_dim=30, tile_size=TILE_SIZE)

    if _own_pool:
        rlrs_pool = queue.Queue()
        for _ in range(num_workers):
            rlrs_pool.put(make_rlrs_instance())

    # --- Prepare Initial Clustering ---
    print("\n========== Initial Clustering ==========")
    initial_result = depp.predict(
        image=image_path,
        num_clusters=num_total_puzzles,
        return_tiles=True
    )

    all_features     = np.array(initial_result["features"])
    current_clusters = build_cluster_data_from_result(initial_result)
    final_solved_images = []

    # =========================
    # Iterative Refine
    # =========================
    prev_assignment_hash = None
    no_change_streak     = 0

    for step in range(max_refine_steps):
        print(f"\n{'=' * 25} Iteration {step} {'=' * 25}")

        original_index_board_for_unsolved = []

        if not current_clusters:
            print("[INFO] No remaining clusters to solve.")
            break

        unsolved_cluster_ids = sorted(current_clusters.keys())
        print(f"[INFO] Solving {len(unsolved_cluster_ids)} clusters with {num_workers} workers...")

        rlrs_feedback          = {}
        current_step_unsolved_imgs = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            def _submit(cid):
                rlrs = rlrs_pool.get()
                try:
                    return solve_single_cluster(rlrs, cid, current_clusters[cid], cid, rl_max_steps)
                finally:
                    rlrs_pool.put(rlrs)

            futures = {executor.submit(_submit, cid): cid for cid in unsolved_cluster_ids}

            for future in as_completed(futures):
                cid = futures[future]
                try:
                    output = future.result()
                except Exception as e:
                    print(f"[ERROR] Cluster {cid} failed: {e}")
                    continue

                score       = output["score"]
                local_score = output["local_score"]
                best_board  = output["best_board"]

                original_indices_for_board = [
                    output["feature_indices"][int(best_board[pos])]
                    for pos in range(len(best_board))
                ]

                if score >= success_threshold:
                    print(f"  Cluster {cid}: [SUCCESS] global={score:.4f}, local={local_score:.4f}")
                    result_img = output["result_img"]
                    if highlight_wrong_edges:
                        result_img = annotate_incorrect_edges(
                            result_img,
                            output["best_board"],
                            output["feature_indices"],
                            meta,
                        )
                        if save_annotated_cluster_images:
                            save_path = f"{output_prefix}_cluster_{cid}_step_{step}.jpg"
                            cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                            print(f"[INFO] Saved annotated cluster image: {save_path}")
                    final_solved_images.append(result_img)
                    original_indices_for_all_board.append(original_indices_for_board)
                else:
                    print(f"  Cluster {cid}: [FAILED ] global={score:.4f}, local={local_score:.4f}")
                    result_img = output["result_img"]
                    if highlight_wrong_edges:
                        result_img = annotate_incorrect_edges(
                            result_img,
                            output["best_board"],
                            output["feature_indices"],
                            meta,
                        )
                        if save_annotated_cluster_images:
                            save_path = f"{output_prefix}_cluster_{cid}_step_{step}.jpg"
                            cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                            print(f"[INFO] Saved annotated cluster image: {save_path}")
                    current_step_unsolved_imgs.append(result_img)
                    original_index_board_for_unsolved.append(original_indices_for_board)
                    rlrs_feedback[cid] = {
                        "must_link":       output["must_link"],
                        "cannot_link":     output["cannot_link"],
                        "tiles":           output["tiles"],
                        "feature_indices": output["feature_indices"],
                    }

        save_image(final_solved_images + current_step_unsolved_imgs,
                   f"{output_prefix}_step_{step}.jpg")

        if len(final_solved_images) >= num_total_puzzles:
            print("\n🎉 All puzzles solved successfully!")
            break
        if step == max_refine_steps - 1:
            print("[INFO] Reached max refine steps.")
            break
        if not rlrs_feedback:
            print("[INFO] No failed clusters left for reclustering.")
            break

        # --- B. Prepare Reclustering Pool ---
        print("[INFO] Preparing constrained reclustering pool...")
        tiles_for_pool, features_for_pool = [], []
        all_must_links, bad_assignments   = [], []
        current_offset = 0
        feedback_ids   = sorted(rlrs_feedback.keys())
        id_map         = {old_id: new_id for new_id, old_id in enumerate(feedback_ids)}

        for old_id in feedback_ids:
            info                   = rlrs_feedback[old_id]
            cluster_must_link      = info["must_link"]
            forbidden_cluster      = id_map[old_id]

            islands = get_island_chains(len(info["tiles"]), cluster_must_link)
            for i in range(1, len(islands)):
                for o_idx in islands[i]:
                    bad_assignments.append((o_idx + current_offset, forbidden_cluster))

            tiles_for_pool.extend(info["tiles"])
            features_for_pool.extend(all_features[info["feature_indices"]])

            for u, v in cluster_must_link:
                all_must_links.append((u + current_offset, v + current_offset))

            current_offset += len(info["tiles"])

        if not tiles_for_pool:
            print("[WARN] Reclustering pool is empty, stopping.")
            break

        # --- C. Constrained Reclustering ---
        print(f"[INFO] Running constrained reclustering on {len(tiles_for_pool)} tiles...")
        new_result = depp.constrained_recluster(
            features=np.array(features_for_pool),
            tiles=tiles_for_pool,
            num_clusters=len(feedback_ids),
            must_link=all_must_links,
            bad_assignments=bad_assignments,
        )

        # --- D. Rebuild current_clusters ---
        pool_to_global = []
        for old_id in feedback_ids:
            pool_to_global.extend(rlrs_feedback[old_id]["feature_indices"])

        rebuilt_clusters = {}
        for pool_idx, new_label in enumerate(new_result["labels"]):
            new_label = int(new_label)
            if new_label not in rebuilt_clusters:
                rebuilt_clusters[new_label] = {"tiles": [], "feature_indices": []}
            rebuilt_clusters[new_label]["tiles"].append(new_result["tiles"][pool_idx])
            rebuilt_clusters[new_label]["feature_indices"].append(pool_to_global[pool_idx])

        current_clusters = rebuilt_clusters
        print(f"[INFO] Reclustering finished. New cluster count: {len(current_clusters)}")

        # --- Early Stop Check ---
        current_assignment = tuple(tuple(board) if board is not None else None for board in original_indices_for_all_board)
        current_hash = hash(current_assignment)

        if prev_assignment_hash is not None:
            if current_hash == prev_assignment_hash:
                no_change_streak += 1
                print(f"[INFO] No change in board assignments. Streak: {no_change_streak}/{early_stop_patience}")
                if no_change_streak >= early_stop_patience:
                    print(f"[INFO] Early stopping due to no changes for {early_stop_patience} consecutive iterations.")
                    break
            else:
                no_change_streak = 0
                print(f"[INFO] Board assignments changed. Resetting streak to 0.")

        prev_assignment_hash = current_hash

    # --- Return Results ---
    boards = original_indices_for_all_board + original_index_board_for_unsolved
    gt_boards, perfect_acc, neighbor_acc, absolute_acc = accuracy(meta, boards)

    return {
        "boards":            boards,
        "perfect_accuracy":  perfect_acc,
        "neighbor_accuracy": neighbor_acc,
        "absolute_accuracy": absolute_acc,
        "gt_boards":         gt_boards,
    }


# =========================
# Main Entry Point
# =========================
if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="Solve mixed jigsaw puzzles using deep learning and RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve a puzzle with default settings
  python main.py --image_path ./sample/instance_000000/mix.png --meta_path ./sample/instance_000000/meta.json
  
  # Solve with edge highlighting and save annotated images
  python main.py --image_path ./mix.png --meta_path ./meta.json --highlight_wrong_edges --save_annotated
  
  # Custom parameters
  python main.py --image_path ./mix.png --meta_path ./meta.json --max_steps 50 --workers 8 --success_threshold 0.85
        """
    )
    
    parser.add_argument("--image_path", type=str, default="./sample/instance_000000/mix.png",
                       help="Path to mixed puzzle image")
    parser.add_argument("--meta_path", type=str, default="./sample/instance_000000/meta.json",
                       help="Path to metadata JSON file")
    parser.add_argument("--output_prefix", type=str, default="./output/output",
                       help="Prefix for output files")
    parser.add_argument("--max_steps", type=int, default=MAX_REFINE_STEPS,
                       help="Maximum refinement iterations")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                       help="Number of parallel workers")
    parser.add_argument("--rl_steps", type=int, default=RL_MAX_STEPS,
                       help="RL steps per cluster")
    parser.add_argument("--success_threshold", type=float, default=SUCCESS_THRESHOLD,
                       help="Success score threshold (0-1)")
    parser.add_argument("--highlight_wrong_edges", action="store_true",
                       help="Highlight incorrect edges in red")
    parser.add_argument("--save_annotated", action="store_true",
                       help="Save annotated cluster images")
    parser.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE,
                       help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image_path):
        print(f"[ERROR] Image file not found: {args.image_path}")
        exit(1)
    if not os.path.exists(args.meta_path):
        print(f"[ERROR] Meta file not found: {args.meta_path}")
        exit(1)
    
    # Create output directory
    output_dir = Path(args.output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("JIGSAW PUZZLE SOLVER - Mixed Puzzle Resolution")
    print("="*60)
    print(f"Image:            {args.image_path}")
    print(f"Meta:             {args.meta_path}")
    print(f"Output prefix:    {args.output_prefix}")
    print(f"Max iterations:   {args.max_steps}")
    print(f"Workers:          {args.workers}")
    print(f"RL steps:         {args.rl_steps}")
    print(f"Success thresh:   {args.success_threshold:.2f}")
    print(f"Edge highlight:   {args.highlight_wrong_edges}")
    print(f"Save annotated:   {args.save_annotated}")
    print("="*60 + "\n")
    
    # Solve puzzle
    try:
        result = solve_puzzle(
            image_path=args.image_path,
            meta_path=args.meta_path,
            output_prefix=args.output_prefix,
            max_refine_steps=args.max_steps,
            num_workers=args.workers,
            rl_max_steps=args.rl_steps,
            success_threshold=args.success_threshold,
            highlight_wrong_edges=args.highlight_wrong_edges,
            save_annotated_cluster_images=args.save_annotated,
            early_stop_patience=args.early_stop_patience,
        )
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Perfect Accuracy:   {result['perfect_accuracy']:.4f} ({result['perfect_accuracy']*100:.2f}%)")
        print(f"Neighbor Accuracy:  {result['neighbor_accuracy']:.4f} ({result['neighbor_accuracy']*100:.2f}%)")
        print(f"Absolute Accuracy:  {result['absolute_accuracy']:.4f} ({result['absolute_accuracy']*100:.2f}%)")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n[WARNING] Process interrupted by user")
        exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    