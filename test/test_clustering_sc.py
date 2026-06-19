import numpy as np
from sklearn.cluster import SpectralClustering
from torchvision import models
import torch.nn as nn
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm import tqdm 
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def split_puzzle_to_tiles(img_path, tile_size=96):
    """
    Splits a puzzle image into its constituent tiles.
    """
    # Read the image and convert to RGB
    img = Image.open(img_path).convert("RGB")
    width, height = img.size

    # Calculate the number of tiles in each dimension
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size
    
    tiles = []
    # Split into num_tiles_x * num_tiles_y blocks
    for row in range(num_tiles_y):
        for col in range(num_tiles_x):
            left = col * tile_size
            upper = row * tile_size
            right = left + tile_size
            lower = upper + tile_size
            tile = img.crop((left, upper, right, lower))
            tiles.append(tile)

    return tiles

def keep_top_k(similarity_matrix, k=1):
        threshold = np.partition(similarity_matrix, -k, axis=1)[:, -k][:, None]
        return np.where(similarity_matrix >= threshold, similarity_matrix, 0)
    
def get_largest(row):
        row_copy = row.copy()
        non_zero = row_copy[row_copy != 0]
        if len(non_zero) < 1:
            return None
        sorted_indices = np.argsort(row_copy)
        top_value = row_copy[sorted_indices[-1]]
        top_index = sorted_indices[-1]
        return top_value, top_index

def resolve_conflicts(similarity_matrix, similarity_matrix_top):
        for _ in range(similarity_matrix.shape[0]):
            stop_flag = True
            for i in range(similarity_matrix.shape[0]):
                for j in range(similarity_matrix.shape[1]):
                    if similarity_matrix_top[i, j] == 0:
                        continue
                    for k in range(similarity_matrix.shape[0]):
                        if k == i or similarity_matrix_top[k, j] == 0:
                            continue
                        if similarity_matrix_top[k, j] >= similarity_matrix_top[i, j]:
                            similarity_matrix_top[i, j] = 0
                            similarity_matrix[i, j] = 0
                            second_largest = get_largest(similarity_matrix[i])
                            if second_largest is not None:
                                second_index = second_largest[1]
                                similarity_matrix_top[i, second_index] = second_largest[0]
                            stop_flag = False
                            break
                        elif similarity_matrix_top[k, j] < similarity_matrix_top[i, j]:
                            similarity_matrix_top[k, j] = 0
                            similarity_matrix[k, j] = 0
                            second_largest = get_largest(similarity_matrix[k])
                            if second_largest is not None:
                                second_index = second_largest[1]
                                similarity_matrix_top[k, second_index] = second_largest[0]
                            stop_flag = False
                            break
                    break
            if stop_flag:
                break
        return similarity_matrix_top
    
from pathlib import Path
import json
def load_meta(json_path):
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        meta = json.load(f)
    return meta

def get_gt_labels(meta):
    labels = [p["group_id"] for p in meta["pieces"]]
    return np.array(labels, dtype=np.int32)

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

class ComboNet(nn.Module):
    def __init__(self, input_shape=(96, 96, 3)):
        super(ComboNet, self).__init__()
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feature_extractor_backbone = efficientnet.features
        num_features = 1280
        self.fen = nn.Sequential(
            feature_extractor_backbone,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self, img1, img2):
        f1_feature = self.fen(img1)
        f2_feature = self.fen(img2)
        concatted_feature = torch.cat((f1_feature, f2_feature), dim=1)
        output = self.classifier_head(concatted_feature)
        return output
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_HORI_PATH = "./model/hori_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth"
MODEL_VRTI_PATH = "./model/vrti_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth"
IMAGE_PATH = "./dataset/30_mix/test/instance_000002/mix.png"
puzzle_meta = "./dataset/30_mix/test/instance_000002/meta.json"
num_clusters = 30
model_hori = ComboNet().to(DEVICE)
model_hori.load_state_dict(torch.load(MODEL_HORI_PATH, map_location=DEVICE))
model_vrti = ComboNet().to(DEVICE)
model_vrti.load_state_dict(torch.load(MODEL_VRTI_PATH, map_location=DEVICE))
model_hori.eval()
model_vrti.eval()

tiles = split_puzzle_to_tiles(IMAGE_PATH, tile_size=96)
total_tiles = len(tiles)

similarity_matrix_hori = np.zeros((total_tiles, total_tiles), dtype=np.float32)
similarity_matrix_vrti = np.zeros((total_tiles, total_tiles), dtype=np.float32)

preprocess = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
tensor_tiles = [preprocess(tile).unsqueeze(0).to(DEVICE) for tile in tiles]
        
pairs1, pairs2, idx_pairs = [], [], []
for i in tqdm(range(total_tiles), desc="Calculating similarity matrix"):
    for j in range(total_tiles):
        if i == j:
            continue
        pairs1.append(tensor_tiles[i])
        pairs2.append(tensor_tiles[j])
        idx_pairs.append((i, j))

        if len(pairs1) >= 256:
            batch1 = torch.cat(pairs1, 0)
            batch2 = torch.cat(pairs2, 0)
            with torch.no_grad():
                probs_h = model_hori(batch1, batch2).cpu().numpy()
                probs_v = model_vrti(batch1, batch2).cpu().numpy()
            for (ii, jj), ph, pv in zip(idx_pairs, probs_h, probs_v):
                similarity_matrix_hori[ii, jj] = ph
                similarity_matrix_vrti[ii, jj] = pv
            pairs1, pairs2, idx_pairs = [], [], []
        
if pairs1:  # flush last batch
    batch1 = torch.cat(pairs1, 0)
    batch2 = torch.cat(pairs2, 0)
    with torch.no_grad():
        probs_h = model_hori(batch1, batch2).cpu().numpy()
        probs_v = model_vrti(batch1, batch2).cpu().numpy()
    for (ii, jj), ph, pv in zip(idx_pairs, probs_h, probs_v):
        similarity_matrix_hori[ii, jj] = ph
        similarity_matrix_vrti[ii, jj] = pv   
        
similarity_matrix_hori_top = keep_top_k(similarity_matrix_hori, k=1)
similarity_matrix_vrti_top = keep_top_k(similarity_matrix_vrti, k=1)

similarity_matrix_hori_top = resolve_conflicts(similarity_matrix_hori, similarity_matrix_hori_top)
similarity_matrix_vrti_top = resolve_conflicts(similarity_matrix_vrti, similarity_matrix_vrti_top)
        
sym_similarity_matrix_hori = (similarity_matrix_hori_top + similarity_matrix_hori_top.T) 
sym_similarity_matrix_vrti = (similarity_matrix_vrti_top + similarity_matrix_vrti_top.T) 
max_similarity_matrix = np.maximum(sym_similarity_matrix_hori, sym_similarity_matrix_vrti)
import random


random_state = random.randint(0, 10000)
clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=random_state
    )
predicted_labels = clustering.fit_predict(max_similarity_matrix)

meta = load_meta(puzzle_meta)
gt_labels = get_gt_labels(meta)
    
aligned_pred, _ = align_clusters(predicted_labels, gt_labels)
perfect_count = count_perfect_groups(aligned_pred, gt_labels)
correct_pieces = count_correctly_clustered_pieces(aligned_pred, gt_labels)
    
print(f"Perfectly clustered groups: {perfect_count} / {num_clusters}")
print(f"Correctly clustered pieces: {correct_pieces} / {num_clusters}")