import json
import os
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import timm
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import concurrent.futures

# Try to import utils
try:
    import utils
except ImportError:
    print("Warning: 'utils' module not found.")

# -----------------------------------------
# 1. CPU-Heavy Task (Must be top-level for pickling)
# -----------------------------------------
from scipy.spatial.distance import cdist # 确保导入这个

def run_cpu_postprocessing(features_numpy, gt_labels, num_clusters, instance_name):
    try:
        # 1. 降维预处理 (SVD)
        F = torch.from_numpy(features_numpy).float()
        M = F - torch.mean(F, dim=0, keepdim=True)
        U, S, V = torch.svd(M)
        U_k = U[:, :num_clusters]
        U_norm = torch.nn.functional.normalize(U_k, p=2, dim=1).numpy()
        
        num_samples = U_norm.shape[0]
        # 计算每个 cluster 应有的理想数量
        target_count = num_samples // num_clusters

        # 2. 获取聚类中心 (初始化)
        # 我们先用普通 KMeans 找到这 num_clusters 个类别的“特征重心”
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        kmeans.fit(U_norm)
        centroids = kmeans.cluster_centers_

        # 3. 计算距离矩阵 (N个碎片 vs K个中心)
        dist_matrix = cdist(U_norm, centroids, metric='euclidean')

        # 4. 基于后悔值的优先级指派 (Balanced Assignment)
        # 后悔值 = 到次优中心的距离 - 到最优中心的距离
        # 后悔值越大，代表该碎片“非它不可”的意愿越强
        sorted_dist = np.sort(dist_matrix, axis=1)
        regret = sorted_dist[:, 1] - sorted_dist[:, 0]
        priority_indices = np.argsort(-regret) # 降序排列

        pred_labels = np.full(num_samples, -1, dtype=np.int32)
        cluster_counts = np.zeros(num_clusters, dtype=np.int32)

        for idx in priority_indices:
            # 该碎片最想去的类别排序
            preferred_clusters = np.argsort(dist_matrix[idx])
            
            assigned = False
            for c_id in preferred_clusters:
                # 如果这个类别还没满，就塞进去
                if cluster_counts[c_id] < target_count:
                    pred_labels[idx] = c_id
                    cluster_counts[c_id] += 1
                    assigned = True
                    break
            
            # 兜底逻辑：如果所有心仪的类都满了（通常发生在总数不能整除时）
            if not assigned:
                # 找当前人数最少的类
                c_id = np.argmin(cluster_counts)
                pred_labels[idx] = c_id
                cluster_counts[c_id] += 1

        # 5. 标签对齐 (Metrics Logic - 保持不变)
        unique_true = np.unique(gt_labels)
        unique_pred = np.unique(pred_labels)
        unique_labels = sorted(list(set(unique_true) | set(unique_pred)))
        
        cm = confusion_matrix(gt_labels, pred_labels, labels=unique_labels)
        row_ind, col_ind = linear_sum_assignment(-cm)
        
        mapping = {unique_labels[c]: unique_labels[r] for r, c in zip(row_ind, col_ind)}
        new_pred = np.array([mapping.get(p, p) for p in pred_labels])

        # 6. 计算得分
        groups = np.unique(gt_labels)
        perfect_count = 0
        for g in groups:
            idx_gt = set(np.where(gt_labels == g)[0])
            idx_pred = set(np.where(new_pred == g)[0])
            if idx_gt == idx_pred:
                perfect_count += 1
        
        correct_pieces = np.sum(new_pred == gt_labels)
        total_pieces = len(gt_labels)

        return {
            "instance": instance_name,
            "perfect": perfect_count,
            "correct": correct_pieces,
            "total": total_pieces,
            "expected_groups": num_clusters
        }
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}", "instance": instance_name}

# -----------------------------------------
# 2. Dataset (Parallel I/O)
# -----------------------------------------
class PuzzleDataset(Dataset):
    def __init__(self, dataset_root, tile_size=96):
        self.dataset_path = Path(dataset_root)
        self.instance_folders = sorted([f for f in self.dataset_path.iterdir() if f.is_dir() and "instance" in f.name])
        self.tile_size = tile_size
        
        # Pre-define transform to avoid re-creating it
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.instance_folders)

    def __getitem__(self, idx):
        folder = self.instance_folders[idx]
        img_path = folder / "mix.png"
        meta_path = folder / "meta.json"

        # Load Meta
        with open(meta_path, "r") as f:
            meta = json.load(f)
        labels = [p["group_id"] for p in meta["pieces"]]
        gt_labels = np.array(labels, dtype=np.int32)

        # Load Image & Split
        # 注意：这里直接切图并转 Tensor，利用 DataLoader 的多线程优势
        # 我们需要把 utils.split_puzzle_into_tiles 的逻辑稍微改写适配 Dataset
        # 假设 utils.split_puzzle_into_tiles 返回的是 list of PIL Images 或 numpy arrays
        raw_tiles = utils.split_puzzle_into_tiles(str(img_path), self.tile_size)
        
        tensor_tiles = []
        for tile in raw_tiles:
            if isinstance(tile, np.ndarray):
                tile = Image.fromarray(tile)
            tensor_tiles.append(self.transform(tile))
            
        # Stack into [N, 3, 224, 224]
        batch_tensor = torch.stack(tensor_tiles)
        
        return batch_tensor, gt_labels, folder.name

def custom_collate(batch):
    # 因为每个拼图的碎片数量(N)可能不同，我们不能使用默认的 default_collate
    # batch_size 我们将设为 1，所以直接取出第一个元素即可
    return batch[0]

# -----------------------------------------
# 3. Model Definition (Same as before)
# -----------------------------------------
class ViTDSCL(nn.Module):
    def __init__(self, embedding_dim=30, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, embedding_dim)
        
    def forward(self, x):
        return self.backbone(x)

# -----------------------------------------
# 4. Main Parallel Evaluation Loop
# -----------------------------------------
def evaluate_parallel(dataset_root, model, device, num_clusters=30, tile_size=96, num_workers=4):
    
    # 1. Setup Data Loader (解决 IO 瓶颈)
    dataset = PuzzleDataset(dataset_root, tile_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                        num_workers=num_workers, collate_fn=custom_collate, 
                        pin_memory=True) # pin_memory 加速 CPU->GPU 传输

    print(f"Dataset size: {len(dataset)} | Workers: {num_workers}")
    
    # Global Metrics
    total_perfect = 0
    total_expected = 0
    total_correct = 0
    total_pieces = 0
    results_log = []

    model.eval()
    
    # 2. Setup Process Pool (解决 CPU 计算瓶颈)
    # max_workers 建议设置为 CPU 核心数 - 2 (留给 GPU 驱动和主线程)
    cpu_workers = max(1, os.cpu_count() - 2)
    print(f"Starting ProcessPoolExecutor with {cpu_workers} workers...")
    
    with torch.no_grad():
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_workers) as executor:
            futures = []
            
            # --- LOOP: GPU Inference (Producer) ---
            print("Starting GPU Inference & Dispatching CPU Tasks...")
            for batch_input, gt_labels, name in tqdm(loader, desc="Inference"):
                batch_input = batch_input.to(device)
                
                # GPU Forward
                features = model(batch_input)
                
                # 将 Tensor 移回 CPU 并转为 Numpy (非阻塞)
                # 只有 Numpy 数组才能安全地跨进程传递
                features_np = features.cpu().numpy()
                
                # Submit task to CPU Pool
                future = executor.submit(run_cpu_postprocessing, 
                                         features_np, 
                                         gt_labels, 
                                         num_clusters, 
                                         name)
                futures.append(future)

            # --- LOOP: Collect Results (Consumer) ---
            print("Waiting for CPU tasks to finish...")
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Post-processing"):
                res = future.result()
                
                if "error" in res:
                    print(f"Error in {res['instance']}: {res['error']}")
                    continue
                
                total_perfect += res["perfect"]
                total_expected += res["expected_groups"]
                total_correct += res["correct"]
                total_pieces += res["total"]
                
                results_log.append(res)

    # --- Final Report (Modified to Write to TXT) ---
    global_acc = total_correct / total_pieces if total_pieces > 0 else 0
    global_pgr = total_perfect / total_expected if total_expected > 0 else 0
    
    # 构造要输出的字符串
    report_content = (
        f"\n{'='*50}\n"
        f"PARALLEL EVALUATION REPORT: {dataset_root}\n"
        f"{'='*50}\n"
        f"Total Instances:      {len(results_log)}\n"
        f"Piece-wise Accuracy:  {global_acc:.4f}\n"
        f"Perfect Group Rate:   {global_pgr:.4f}\n"
        f"{'='*50}\n"
    )

    print(report_content)

    with open("evaluation_log.txt", "a", encoding="utf-8") as f:
        f.write(report_content)

# -----------------------------------------
# 5. Entry Point
# -----------------------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True 
    
    # Config
    DATASET_ROOT = "./DEPP/clustering_dataset/6_mix/test"
    CHECKPOINT = "./model/dscl_vit_epoch_40.pth"
    NUM_CLUSTERS = 6
    TILE_SIZE = 96
    
    # Detect CPU cores
    NUM_WORKERS = 12 # DataLoader workers (IO)
    
    # Model Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTDSCL(embedding_dim=30)
    
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.to(device)
        print("Model Loaded.")
    else:
        print("Checkpoint not found.")
        exit()

    # Run
    """ 
    DATASET_ROOT = "./puzzle_dataset/6_mix/test"
    NUM_CLUSTERS = 6
    evaluate_parallel(DATASET_ROOT, model, device, NUM_CLUSTERS, TILE_SIZE, num_workers=NUM_WORKERS)
    """ 
    DATASET_ROOT = "./puzzle_dataset/12_mix/test"
    NUM_CLUSTERS = 12
    evaluate_parallel(DATASET_ROOT, model, device, NUM_CLUSTERS, TILE_SIZE, num_workers=NUM_WORKERS)
    """ 
    DATASET_ROOT = "./DEPP/clustering_ dataset/20_mix/test"
    NUM_CLUSTERS = 20
    evaluate_parallel(DATASET_ROOT, model, device, NUM_CLUSTERS, TILE_SIZE, num_workers=NUM_WORKERS)
    
    DATASET_ROOT = "./DEPP/clustering_dataset/30_mix/test"
    NUM_CLUSTERS = 30
    evaluate_parallel(DATASET_ROOT, model, device, NUM_CLUSTERS, TILE_SIZE, num_workers=NUM_WORKERS)"""