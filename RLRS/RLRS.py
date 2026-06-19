import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import models, transforms
from stable_baselines3 import DQN
from RLRS.environment_swap import JigsawSwapEnv
from utils import split_puzzle_into_tiles

class ComboNet(nn.Module):
    """CNN model for calculating tile adjacency scores."""
    def __init__(self):
        super(ComboNet, self).__init__()
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.fen = nn.Sequential(
            efficientnet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 512),
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
        f1 = self.fen(img1)
        f2 = self.fen(img2)
        return self.classifier_head(torch.cat((f1, f2), dim=1))

class RLRS:
    def __init__(self, model_hori_path, model_vrti_path, dqn_path, pdn_path, device=None):
        """Initializes and loads all necessary models."""
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CNN models
        self.model_hori = ComboNet().to(self.device)
        self.model_hori.load_state_dict(torch.load(model_hori_path, map_location=self.device))
        self.model_hori.eval()

        self.model_vrti = ComboNet().to(self.device)
        self.model_vrti.load_state_dict(torch.load(model_vrti_path, map_location=self.device))
        self.model_vrti.eval()

        # Load RL model
        self.rl_model = DQN.load(dqn_path)
        
        # Load PDN model
        self.pdn_model = models.efficientnet_b3(weights=None)
        in_features = self.pdn_model.classifier[1].in_features
        self.pdn_model.classifier[1] = torch.nn.Linear(in_features, 2)
        self.pdn_model.load_state_dict(torch.load(pdn_path, map_location="cpu"))
        self.pdn_model.to(self.device)
        self.pdn_model.eval()
        
        self.best_board = None
        self.best_score = -float('inf')
        self.best_local_score = -float('inf')
        self.best_global_score = -float('inf')
        
        self.alpha = 0.6  # Weight for global score in combined scoring

        self.preprocess_tiles = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.preprocess_puzzles = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def _build_similarity_matrices(self, tiles):
        """Precompute S_LR and S_UD matrices."""
        n = len(tiles)
        
        tiles_tensor = torch.from_numpy(np.array(tiles)).permute(0, 3, 1, 2).float() / 255.0
        tiles_tensor = tiles_tensor.to(self.device)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        tiles_tensor = TF.resize(tiles_tensor, [96, 96], antialias=True)
        tensor_tiles = (tiles_tensor - mean) / std
        
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        idx_i, idx_j = [p[0] for p in pairs], [p[1] for p in pairs]

        with torch.no_grad():
            out_hori = self.model_hori(tensor_tiles[idx_i], tensor_tiles[idx_j]).squeeze().cpu().numpy()
            out_vrti = self.model_vrti(tensor_tiles[idx_i], tensor_tiles[idx_j]).squeeze().cpu().numpy()

        S_LR, S_UD = np.zeros((n, n), dtype=np.float32), np.zeros((n, n), dtype=np.float32)
        for k, (i, j) in enumerate(pairs):
            S_LR[i, j] = out_hori[k]
            S_UD[i, j] = out_vrti[k]
        return S_LR, S_UD
    
    def _render_canvas(self, board, tiles, grid_size=3):
        h, w, c = tiles[0].shape
        canvas = np.zeros((grid_size * h, grid_size * w, c), dtype=np.uint8)
        for pos, piece_id in enumerate(board):
            r, c_ = pos // grid_size, pos % grid_size
            canvas[r*h:(r+1)*h, c_*w:(c_+1)*w, :] = tiles[int(piece_id)]
        return canvas
    
    def _calculate_global_score(self, board, tiles, grid_size=3):
        current_canvas = self._render_canvas(board, tiles, grid_size)
        pil_img = Image.fromarray(current_canvas)
        img_tensor = self.preprocess_puzzles(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pdn_out = self.pdn_model(img_tensor)
            global_score = torch.softmax(pdn_out, dim=1)[0][1].item() 
        return global_score
    
    def _is_mutual_best(self, i, j, S_matrix, threshold=0.85, margin=0.15):
        score = S_matrix[i, j]
        
        if score < threshold:
            return False
            
        scores_i_to_others = S_matrix[i, :] 
        scores_others_to_j = S_matrix[:, j]  

        # Mutual Best Check
        if score < np.max(scores_i_to_others) - 1e-5:
            return False 
        if score < np.max(scores_others_to_j) - 1e-5:
            return False 
        return True

    def solve(self, image_path=None, tiles=None, grid_size=3, tile_size=104, gap=4, max_steps=200):
        """Solves the jigsaw puzzle and returns the solved image array."""
        # 1. Prepare Tiles
        if tiles is not None:
            numpy_tiles = []
            for t in tiles:
                if isinstance(t, Image.Image):
                    numpy_tiles.append(np.array(t))
                else:
                    numpy_tiles.append(t)
            tiles = numpy_tiles
        if tiles is None:
            tiles = split_puzzle_into_tiles(image_path, tile_size=tile_size - gap * 2)
        
        # 2. Build Matrices and Env
        S_LR, S_UD = self._build_similarity_matrices(tiles)
        mock_data = [{'image_id': 'solver_task', 'S_LR': S_LR, 'S_UD': S_UD}]
        env = JigsawSwapEnv(mock_data, is_training=False, max_steps=max_steps)
        
        obs, _ = env.reset()
        state_history = {}
        global_score_cache = {}
        self.best_board = env.board.copy()
        self.best_local_score = env._calculate_total_score(env.board) / 12.0
        self.best_global_score = self._calculate_global_score(env.board, tiles, grid_size)
        self.best_score = self.best_global_score * self.alpha + self.best_local_score * (1 - self.alpha)

        step = 0
        # 3. RL Inference Loop
        for _ in range(max_steps):
            board_tuple = tuple(env.board)
            state_history[board_tuple] = state_history.get(board_tuple, 0) + 1
            
            # Escape local optima/cycles
            if state_history[board_tuple] > 2:
                action = env.action_space.sample()
            else:
                action, _ = self.rl_model.predict(obs, deterministic=True)

            obs, _, terminated, truncated, _ = env.step(action)
            step += 1
            
            # Access the global score
            new_board_tuple = tuple(env.board)
            if new_board_tuple in global_score_cache:
                global_score = global_score_cache[new_board_tuple] # cache hit
            else:
                global_score = self._calculate_global_score(env.board, tiles, grid_size)
                global_score_cache[new_board_tuple] = global_score # cache store
            
            # Access the local score
            local_score = env._calculate_total_score(env.board) / 12.0
            
            # Calculate combined score
            total_score = self.alpha * global_score + (1 - self.alpha) * local_score
            
            if total_score > self.best_score:
                self.best_score = total_score
                self.best_local_score = local_score
                self.best_global_score = global_score
                self.best_board = env.board.copy()
                
            if total_score > 0.95:
                self.best_board = env.board.copy() 
                break

            if terminated or truncated:
                break

        # 4. Reconstruct
        # =================================================
        must_link = []
        cannot_link = []
        LOW_THRES = 0.10  # Threshold for strong negative evidence

        for r in range(grid_size):
            for c in range(grid_size):
                pos = r * grid_size + c
                current_piece = int(self.best_board[pos])
                
                # check right neighbor (LR)
                if c < grid_size - 1:
                    right_piece = int(self.best_board[pos + 1])
                    if self._is_mutual_best(current_piece, right_piece, S_LR, threshold=0.85, margin=0.15):
                        must_link.append((current_piece, right_piece))
                    elif S_LR[current_piece, right_piece] < LOW_THRES:
                        cannot_link.append((current_piece, right_piece))
                        
                # check down neighbor (UD)
                if r < grid_size - 1:
                    down_piece = int(self.best_board[pos + grid_size])
                    if self._is_mutual_best(current_piece, down_piece, S_UD, threshold=0.85, margin=0.15):
                        must_link.append((current_piece, down_piece))
                    elif S_UD[current_piece, down_piece] < LOW_THRES:
                        cannot_link.append((current_piece, down_piece))
        # =================================================
        
        h, w, c = tiles[0].shape
        canvas = np.zeros((grid_size * h, grid_size * w, c), dtype=np.uint8)
        for pos, piece_id in enumerate(self.best_board):
            r, c_ = pos // grid_size, pos % grid_size
            canvas[r*h:(r+1)*h, c_*w:(c_+1)*w, :] = tiles[int(piece_id)]
        
        print(step)
        return canvas, must_link, cannot_link, self.best_board, S_LR, S_UD
    