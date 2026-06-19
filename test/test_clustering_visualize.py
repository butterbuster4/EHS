import json
from pathlib import Path
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import timm
import utils
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

# -----------------------------------------
# 1. Model Definition: ViT-DSCL
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
# 2. Spectral Clustering Inference
# -----------------------------------------
def spectral_clustering_inference(F, k):
    if isinstance(F, np.ndarray):
        F = torch.from_numpy(F)
    F = F.float()
    
    M = F - torch.mean(F, dim=0, keepdim=True)
    U, S, V = torch.svd(M)
    U_k = U[:, :k] 
    U_norm = torch.nn.functional.normalize(U_k, p=2, dim=1)
    features_for_kmeans = U_norm.detach().cpu().numpy()
    
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    pred_labels = kmeans.fit_predict(features_for_kmeans)
    
    return pred_labels

# -----------------------------------------
# 3. Predict Puzzle Groups
# -----------------------------------------
def predict_puzzle_groups(model, image_list, num_clusters=30, device='cuda'):
    model.eval()
    
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    batch_tensors = []
    for img in image_list:
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        tensor = inference_transform(img)
        batch_tensors.append(tensor)
        
    batch_input = torch.stack(batch_tensors).to(device)
    
    with torch.no_grad():
        F = model(batch_input)
    
    predicted_labels = spectral_clustering_inference(F.cpu(), k=num_clusters)
    
    return predicted_labels

def get_gt_labels(meta):
    labels = [p["group_id"] for p in meta["pieces"]]
    return np.array(labels, dtype=np.int32)

def load_meta(json_path):
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        meta = json.load(f)
    return meta

def align_clusters(pred_labels, true_labels):
    """
    Align predicted clusters to ground truth clusters using Hungarian matching.
    Robust to non-consecutive labels and subset matching.
    """
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)
    
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)
    
    unique_labels = sorted(list(set(unique_true) | set(unique_pred)))
    
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)

    row_ind, col_ind = linear_sum_assignment(-cm)
    
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        gt_val = unique_labels[r]
        pred_val = unique_labels[c]
        mapping[pred_val] = gt_val
        
    def map_func(p):
        return mapping.get(p, p)
        
    new_pred = np.array([map_func(p) for p in pred_labels])
    
    return new_pred, mapping

def count_perfect_groups(aligned, gt):
    groups = np.unique(gt)
    perfect = 0
    
    for g in groups:
        idx_gt = set(np.where(gt == g)[0])
        
        idx_pred = set(np.where(aligned == g)[0])
        
        if idx_gt == idx_pred:
            perfect += 1
            
    return perfect

def count_correctly_clustered_pieces(aligned, gt):
    return np.sum(aligned == gt)

# -----------------------------------------
# 4. Sample Usage
# -----------------------------------------
if __name__ == "__main__":
    instance_path = "./instance_000000"
    #instance_path = "./test/instance_000000"
    puzzle = instance_path + "/mix.png"
    puzzle_meta = instance_path + "/meta.json"
    checkpoint_path = "./model/dscl_vit_epoch_40.pth"
    #checkpoint_path = "./model/best_model.pth"
    tile_size = 96
    
    # A. Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTDSCL(embedding_dim=30, pretrained=False) 
    
    # Load trained weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    # B. Split puzzle into tiles
    tiles = utils.split_puzzle_into_tiles(puzzle, tile_size)
    
    # C. Predict puzzle groups
    meta = load_meta(puzzle_meta)
    gt_labels = get_gt_labels(meta)
    num_clusters = len(np.unique(gt_labels))
    predicted_labels = predict_puzzle_groups(model, tiles, num_clusters=num_clusters, device=device)
    
    aligned_pred, _ = align_clusters(predicted_labels, gt_labels)
    perfect_count = count_perfect_groups(aligned_pred, gt_labels)
    correct_pieces = count_correctly_clustered_pieces(aligned_pred, gt_labels)
    
    print(f"Perfectly clustered groups: {perfect_count} / {num_clusters}")
    print(f"Correctly clustered pieces: {correct_pieces} / {num_clusters * 9}")
    
    # D. Visualize the clustering results and save to file
    clusters = utils.split_tiles_according_to_labels(tiles, aligned_pred)
    save_path = "./result_clusters.png"
    utils.show_clusters_multi_grid(clusters, save_path=save_path, tile_size=tile_size, max_cols=3, cluster_cols=3)
    
    
    