import os, random
import shutil
from pathlib import Path
from tqdm import tqdm

def split_train_and_test(data_dir, out_dir, train_ratio=0.9, seed=42):
    train_dir = Path(out_dir) / 'train'
    test_dir = Path(out_dir) / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    imgs = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    random.seed(seed)
    random.shuffle(imgs)
    
    n_train = int(len(imgs) * train_ratio)
    train_imgs = imgs[:n_train]
    test_imgs = imgs[n_train:]

    for img in tqdm(train_imgs):
        src_path = Path(data_dir) / img
        dst_path = train_dir / img
        shutil.copy(src_path, dst_path)
        
    for img in tqdm(test_imgs):
        src_path = Path(data_dir) / img
        dst_path = test_dir / img
        shutil.copy(src_path, dst_path)
        
    print("Data split completed.")
    
if __name__ == "__main__":
    split_train_and_test(
        data_dir="./met_raw_data",    
        out_dir="./dataset",    
        train_ratio=0.9
    )