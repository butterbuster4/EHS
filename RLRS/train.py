import numpy as np
import torch
import torch.nn as nn
import pickle
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment_swap import JigsawSwapEnv 

# =====================================================================
# 🧠 Core Surgery Area: Custom 2D CNN Brain (Custom Feature Extractor)
# =====================================================================
class JigsawCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, grid_size=3, embed_dim=32):
        # observation_space.shape[0] should be 171
        super(JigsawCNNFeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        
        # 1. Word embedding layer for puzzle IDs (converts 0-8 categorical variables to 32-dimensional dense vectors)
        self.embedding = nn.Embedding(num_embeddings=grid_size**2, embedding_dim=embed_dim)
        
        # 2. Spatial topology extractor (2D CNN)
        # Input dimension: embed_dim (32), Output dimension: 64, Convolution kernel 3x3 (exactly covers the 3x3 puzzle board)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten() # Flattened output dimension: 128 * 3 * 3 = 1152
        )
        
        # 3. Pre-calculated matrix processing branch (MLP)
        # Assuming that 171 dimensions consist of 9 for the board and 162 for S_LR and S_UD matrices
        self.matrix_mlp = nn.Sequential(
            nn.Linear(162, 128),
            nn.ReLU()
        )
        
        # 4. Feature fusion layer
        # Combine CNN extracted "spatial intuition" (1152) with MLP extracted "local edge scores" (128)
        concat_dim = 1152 + 128
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # ⚠️ Critical data slicing: Ensure that the index here aligns with the format of the observation output from your env.step()!
        # Assuming the first 9 positions are current_board and the remaining 162 are S_LR and S_UD matrices flattened
        board_obs = observations[:, :9].long()   # Extract the first 9 puzzle piece IDs, convert to integer
        matrix_obs = observations[:, 9:].float() # Extract the remaining 162 float values representing edge scores

        # --- Branch 1: Process Board spatial topology ---
        # 1. Convert to Embedding: shape -> (batch_size, 9, 32)
        embedded_board = self.embedding(board_obs)
        
        # 2. Reshape to 2D image format (batch_size, channels, height, width) -> (batch_size, 32, 3, 3)
        board_2d = embedded_board.view(-1, self.grid_size, self.grid_size, self.embed_dim).permute(0, 3, 1, 2)
        
        # 3. Extract spatial features through CNN -> (batch_size, 1152)
        cnn_features = self.cnn(board_2d)

        # --- Branch 2: Process pre-calculated score matrices ---
        # (batch_size, 128)
        matrix_features = self.matrix_mlp(matrix_obs)

        # --- Branch 3: Brain fusion ---
        # Merge spatial intuition and edge scores -> (batch_size, features_dim)
        combined_features = torch.cat((cnn_features, matrix_features), dim=1)
        
        return self.fusion(combined_features)


# =====================================================================
# 🚀 Main Training Process
# =====================================================================

# 1. Load your dataset 
with open("./reassembling_dataset/train_dataset_3x3_4gap.pkl", "rb") as f: dataset_train = pickle.load(f)
with open("./reassembling_dataset/test_dataset_3x3_4gap.pkl", "rb") as f: dataset_val = pickle.load(f)

# 2. Instantiate the environment and wrap it with Monitor
env = JigsawSwapEnv(dataset=dataset_train, is_training=True)
env = Monitor(env)

check_env(env, warn=True)
print("Environment check passed! Ready to start training...")

# 3. Set up the neural network architecture with a custom “visual brain”
# Note: We need to tell SB3 to use our JigsawCNNFeatureExtractor
policy_kwargs = dict(
    features_extractor_class=JigsawCNNFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=256), # Core feature dimension extracted
    net_arch=[256, 128] # The action selection network (Q-Network) doesn't need to be as deep since the features are already perfect
)

# 4. Initialize the DQN model (still write MlpPolicy, SB3 will automatically replace the earlier layers with the feature extractor)
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=128,
    gamma=0.99,
    exploration_initial_eps=1.0,
    exploration_fraction=0.70,
    target_update_interval=500,
    policy_kwargs=policy_kwargs, # 💥 Inject the new 2D CNN brain
    tensorboard_log="./jigsaw_tensorboard/",
    verbose=1,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 5. Optional: Set up evaluation callback
eval_env = Monitor(JigsawSwapEnv(dataset=dataset_val, is_training=True))
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path='./logs/best_model/',
    log_path='./logs/results/', 
    eval_freq=5000,
    deterministic=True, 
    render=False
)

# 6. Start training!
print("Start training with 2D Spatial Brain...")
model.learn(total_timesteps=2000000, callback=eval_callback, progress_bar=True)

# 7. Save the final model
model.save("dqn_jigsaw_spatial_final")
print("Training complete and saved!")