import pickle
import numpy as np
from tqdm import tqdm
from RLRS.environment_swap import JigsawSwapEnv
from RLRS.RLRS import RLRS

def calculate_perfect_accuracy(board):
    if all(board[i] == i for i in range(9)):
        return 1
    return 0

def calculate_absolute_accuracy(board):
    return sum(1 for i in range(9) if board[i] == i) 

def calculate_neighbour_accuracy(board):
    correct_edges = 0
    for pos in range(9):
        piece_id = board[pos]
        if piece_id == -1:
            continue
        r = pos // 3
        c = pos % 3
        
        if c < 2: 
            right_neighbor_pos = r * 3 + (c + 1)
            if right_neighbor_pos < 9 and board[right_neighbor_pos] != -1:
                if piece_id + 1 == board[right_neighbor_pos]:
                    correct_edges += 1
        
        if r < 2: 
            down_neighbor_pos = (r + 1) * 3 + c
            if down_neighbor_pos < 9 and board[down_neighbor_pos] != -1:
                if piece_id + 3 == board[down_neighbor_pos]:
                    correct_edges += 1
    return correct_edges
    

solver = RLRS(
    model_hori_path="./model/hori_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth",
    model_vrti_path="./model/vrti_4_EfficientNetB0_25class_ft_006_with_sigmoid.pth",
    dqn_path="./model/dqn_jigsaw_final.zip",
    pdn_path="./model/best_efficientnetb3.pth"
)

# 2. Load the pre-computed similarity dataset
TEST_DATASET_PATH = "./RLRS/reassembling_dataset/test_dataset_3x3_4gap.pkl"
with open(TEST_DATASET_PATH, 'rb') as f:
    test_data = pickle.load(f)

# 3. Create the evaluation environment using the loaded data
# is_training=False ensures deterministic behavior
env = JigsawSwapEnv(test_data, is_training=False, max_steps=60)

perfect_count = 0
absolute_count = 0
neighbour_count = 0
total_instances = len(test_data)

print(f"Starting batch evaluation on {total_instances} instances...")

for i in tqdm(range(total_instances)):
    # Reset to a specific image instance from the dataset
    obs, _ = env.reset() 
    
    # Note: If your dataset starts with a perfect board, 
    # you MUST shuffle it manually to test the solver's ability.
    np.random.shuffle(env.board)
    obs = env._get_obs() # Refresh observation after shuffle

    state_history = {}
    done = False
    
    # Inference Loop
    while not done:
        board_tuple = tuple(env.board)
        state_history[board_tuple] = state_history.get(board_tuple, 0) + 1
        
        # Cycle breaking logic
        if state_history[board_tuple] > 2:
            action = env.action_space.sample()
        else:
            # Use the DQN model inside the solver
            action, _ = solver.rl_model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            if truncated and not terminated:
                print(f"⚠️  Instance {i} reached max steps without solving.")
                print(f"   Final Board: {env.board}, Score: {env.score:.4f}, image_id: {env.current_image_id}")
            perfect_count += calculate_perfect_accuracy(env.board)
            absolute_count += calculate_absolute_accuracy(env.board)
            neighbour_count += calculate_neighbour_accuracy(env.board)
            done = True

# 4. Final Statistics
perfect_accuracy = (perfect_count / total_instances) * 100
absolute_accuracy = (absolute_count / (total_instances * 9)) * 100
neighbour_accuracy = (neighbour_count / (total_instances * 12)) * 100
print(f"\nEvaluation Finished!")
print(f"Total Images: {total_instances}")
print(f"Perfectly Solved: {perfect_count}")
print(f"Perfect Accuracy: {perfect_accuracy:.2f}%")
print(f"Absolute Accuracy: {absolute_accuracy:.2f}%")
print(f"Neighbour Accuracy: {neighbour_accuracy:.2f}%")