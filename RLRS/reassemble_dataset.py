import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm 
from utils import extract_tiles_from_image

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

def calculate_total_score(adjacency_matrix_hori, adjacency_matrix_vrti):
    total_score = 0
    num_tiles = adjacency_matrix_hori.shape[0]
    for i in range(num_tiles):
        if i % 3 != 2:  # not the last column
            total_score += adjacency_matrix_hori[i, i + 1]
        if i < 6:  # not the last row        
            total_score += adjacency_matrix_vrti[i, i + 3]
    return total_score

class PuzzleDataset(Dataset):
    def __init__(self, folder_path, preprocess, tile_size=104, gap=4):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.preprocess = preprocess
        self.tile_size = tile_size
        self.gap = gap

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        
        tiles = extract_tiles_from_image(img_path, tile_per_side=3, tile_size=self.tile_size, gap=self.gap)
        
        if tiles is None or len(tiles) != 9:
            return torch.zeros(0), img_name
            
        tensor_tiles = torch.stack([self.preprocess(Image.fromarray(tile)) for tile in tiles])
        return tensor_tiles, img_name

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current running device: {DEVICE}")
    
    # Path configuration
    MODEL_HORI_PATH = "./model/hori_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth"
    MODEL_VRTI_PATH = "./model/vrti_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth"
    TRAIN_DATASET_PATH = "E:/MET_dataset/test" 
    OUTPUT_PKL_PATH = "./test.pkl"   
    
    # Preprocessing pipeline for the tiles
    preprocess = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # load models
    model_hori = ComboNet().to(DEVICE)
    model_hori.load_state_dict(torch.load(MODEL_HORI_PATH, map_location=DEVICE))
    model_hori.eval()
    
    model_vrti = ComboNet().to(DEVICE)
    model_vrti.load_state_dict(torch.load(MODEL_VRTI_PATH, map_location=DEVICE))
    model_vrti.eval()

    # dataset and dataloader
    dataset = PuzzleDataset(TRAIN_DATASET_PATH, preprocess)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=min(4, os.cpu_count()))
    
    valid_pairs = [(i, j) for i in range(9) for j in range(9) if i != j]
    idx_i = [p[0] for p in valid_pairs]
    idx_j = [p[1] for p in valid_pairs]

    final_dataset_list = []
    
    print(f"Processing {len(dataset)} images from {TRAIN_DATASET_PATH}...")
    for tensor_tiles_batch, img_name_batch in tqdm(dataloader):
        tensor_tiles = tensor_tiles_batch.squeeze(0) 
        img_name = img_name_batch[0]
        
        if tensor_tiles.nelement() == 0:
            continue
            
        tensor_tiles = tensor_tiles.to(DEVICE)
        
        # Batch inference for all pairs
        img1_batch = tensor_tiles[idx_i]  # Shape: (72, 3, 96, 96)
        img2_batch = tensor_tiles[idx_j]  # Shape: (72, 3, 96, 96)
        
        with torch.no_grad():
            output_hori = model_hori(img1_batch, img2_batch).squeeze() # Shape: (72,)
            output_vrti = model_vrti(img1_batch, img2_batch).squeeze() # Shape: (72,)
            
        out_hori_np = output_hori.cpu().numpy()
        out_vrti_np = output_vrti.cpu().numpy()
        
        adj_matrix_hori = np.zeros((9, 9), dtype=np.float32)
        adj_matrix_vrti = np.zeros((9, 9), dtype=np.float32)
        
        adj_matrix_hori[idx_i, idx_j] = out_hori_np
        adj_matrix_vrti[idx_i, idx_j] = out_vrti_np
        
        perfect_score = calculate_total_score(adj_matrix_hori, adj_matrix_vrti)
        
        instance = {
            'S_LR': adj_matrix_hori,
            'S_UD': adj_matrix_vrti,
            'image_id': img_name,
            'perfect_score': float(perfect_score)
        }
        final_dataset_list.append(instance)

    print(f"💾 saving processed data to {OUTPUT_PKL_PATH}...")
    with open(OUTPUT_PKL_PATH, 'wb') as f:
        pickle.dump(final_dataset_list, f)
        