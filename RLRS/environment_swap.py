import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools

class JigsawSwapEnv(gym.Env):
    def __init__(self, dataset, is_training=False, max_steps=100):
        super().__init__()
        
        self.dataset = dataset
        self.is_training = is_training
        self.max_steps = max_steps
        self.board = None
        self.base_S_LR = None
        self.base_S_UD = None
        self.steps = 0
        self.score = 0.0
        
        self.total_env_steps = 0 
        
        # Action space
        self.pairs = list(itertools.combinations(range(9), 2))
        self.action_space = spaces.Discrete(len(self.pairs))
        
        # State space: 9x9 S_LR + 9x9 S_UD
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(162,),
            dtype=np.float32
        )
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # 1. randomly select an instance
        idx = np.random.randint(0, len(self.dataset))
        data_instance = self.dataset[idx]
        
        self.base_S_LR = data_instance['S_LR']
        self.base_S_UD = data_instance['S_UD']
        self.current_image_id = data_instance['image_id']
        self.current_perfect_score = data_instance.get('perfect_score', 0.0)
        
        self.board = np.arange(9)
        
        if self.is_training:
            if self.total_env_steps < 300000:
                scramble_steps = 1
            elif self.total_env_steps < 800000:
                scramble_steps = 2
            elif self.total_env_steps < 1400000:
                scramble_steps = 4
            else:
                scramble_steps = -1  
                
            if scramble_steps == -1:
                np.random.shuffle(self.board)
            else:
                for _ in range(scramble_steps):
                    idx1, idx2 = np.random.choice(9, 2, replace=False)
                    self.board[idx1], self.board[idx2] = self.board[idx2], self.board[idx1]
        
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        self.total_env_steps += 1
        i, j = self.pairs[action]
        
        old_correct_positions = np.sum(self.board == np.arange(9))
        old_total_score = self._calculate_total_score(self.board)
        
        self.board[i], self.board[j] = self.board[j], self.board[i]
        
        new_correct_positions = np.sum(self.board == np.arange(9))
        new_total_score = self._calculate_total_score(self.board)
        
        reward_relative = new_total_score - old_total_score
        reward_absolute = (new_correct_positions - old_correct_positions) * 1.0
        
        reward = reward_relative + reward_absolute - 0.1
            
        """if self.is_training:
            terminated = bool(new_correct_positions == 9)
        else:
            terminated = False"""
        terminated = bool(new_correct_positions == 9)
        if terminated:
            reward += 10.0
        truncated = bool(self.steps >= self.max_steps)
        
        self.score = new_total_score
        return self._get_obs(), float(reward), terminated, truncated, {}
            
    def _get_obs(self):
        current_S_LR = self.base_S_LR[self.board][:, self.board]
        current_S_UD = self.base_S_UD[self.board][:, self.board]
        
        return np.concatenate([
            current_S_LR.flatten(),
            current_S_UD.flatten()
        ])
        
    """def _get_obs(self):
        current_S_LR = self.base_S_LR[self.board][:, self.board]
        current_S_UD = self.base_S_UD[self.board][:, self.board]
        
        # 💥 核心修改：在最前面加入 self.board
        return np.concatenate([
            self.board.astype(np.float32), # <-- 新增这一行，占 9 个维度
            current_S_LR.flatten(),
            current_S_UD.flatten()
        ])"""
        
    def _calculate_total_score(self, board):
        total_score = 0.0
        for pos in range(9):
            piece_id = board[pos]
            if piece_id == -1:
                continue
            r = pos // 3
            c = pos % 3
            
            if c < 2: 
                right_neighbor_pos = r * 3 + (c + 1)
                if right_neighbor_pos < 9 and board[right_neighbor_pos] != -1:
                    total_score += self.base_S_LR[piece_id, board[right_neighbor_pos]]
            
            if r < 2: 
                down_neighbor_pos = (r + 1) * 3 + c
                if down_neighbor_pos < 9 and board[down_neighbor_pos] != -1:
                    total_score += self.base_S_UD[piece_id, board[down_neighbor_pos]]
        
        return total_score