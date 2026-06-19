import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from environment_swap import JigsawSwapEnv 

# 1. Load dataset
import pickle
with open("./reassembling_dataset/train_dataset_3x3_4gap.pkl", "rb") as f: dataset_train = pickle.load(f)
with open("./reassembling_dataset/test_dataset_3x3_4gap.pkl", "rb") as f: dataset_val = pickle.load(f)
# --------------------------------------------------------

# 2. Create environment
env = JigsawSwapEnv(dataset=dataset_train, is_training=True)
env = Monitor(env) 

# Evaluate the environment to check if it's compatible with Stable Baselines3
check_env(env, warn=True)
print("Environment check passed! Ready to start training...")

# 3. Set up DQN policy kwargs (custom network architecture)
policy_kwargs = dict(net_arch=[512, 256, 128])

# 4. Initialize DQN agent
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,          # Experience replay buffer size
    learning_starts=10000,        # The first 10000 steps are pure random exploration, collecting initial data
    batch_size=128,              # Batch size for each sample
    gamma=0.99,                  # Discount factor
    exploration_initial_eps=1.0,  # Initially completely random
    exploration_fraction=0.70,    # Use the first 30% of training time for decaying epsilon exploration rate
    target_update_interval=500,  # Frequency of target network updates
    policy_kwargs=policy_kwargs, # Inject custom network architecture
    tensorboard_log="./jigsaw_tensorboard/", # TensorBoard log directory
    verbose=1,
    device="cuda"                # Use GPU if available
)

# 5. Optional: Set up evaluation callback (test on validation set while training)
eval_env = Monitor(JigsawSwapEnv(dataset=dataset_val, is_training=True))
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path='./logs/best_model/',
    log_path='./logs/results/', 
    eval_freq=5000, # Evaluate every 5000 steps
    deterministic=True, 
    render=False
)

# 6. Start training!
print("Start training...")
model.learn(total_timesteps=2000000, callback=eval_callback, progress_bar=True)

# 7. Save the final model
model.save("dqn_jigsaw_final")
print("Training complete and saved!")