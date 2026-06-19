import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import timm
from pathlib import Path
import cv2
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import networkx as nx
from scipy.spatial.distance import cdist

class ViTDSCL(nn.Module):
    def __init__(self, embedding_dim=30, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, embedding_dim)
        
    def forward(self, x):
        return self.backbone(x)


class DEPP:    
    def __init__(self, checkpoint_path, embedding_dim=30, tile_size=96, device=None):
        self.tile_size = tile_size
        self.embedding_dim = embedding_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTDSCL(embedding_dim=embedding_dim, pretrained=False)
        
        # Load pre-trained weights
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"DEPP initialized successfully. Device: {self.device}")
    
    def split_image_into_tiles(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to numpy for splitting
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        tiles = []
        positions = []  # Record the position information of each tile
        
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                tile = img_array[y:y+self.tile_size, x:x+self.tile_size]
                tiles.append(Image.fromarray(tile))
                positions.append((y, x))
        
        return tiles, positions
    
    def extract_features(self, tiles):
        # Convert to tensor
        batch_tensors = []
        for tile in tiles:
            if isinstance(tile, np.ndarray):
                tile = Image.fromarray(tile)
            batch_tensors.append(self.transform(tile))
        batch_input = torch.stack(batch_tensors).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch_input)
        
        return features.cpu().numpy()

    def spectral_clustering(self, features, num_clusters):
        if features is None or len(features) == 0:
            raise ValueError("features is empty in spectral_clustering.")

        features = np.asarray(features, dtype=np.float32)
        num_samples = features.shape[0]

        if num_clusters <= 0:
            raise ValueError(f"num_clusters must be positive, got {num_clusters}.")

        if num_clusters > num_samples:
            num_clusters = num_samples

        # 1. Centering + SVD
        F = torch.from_numpy(features).float()
        M = F - torch.mean(F, dim=0, keepdim=True)

        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        k = min(num_clusters, U.shape[1])
        U_k = U[:, :k]
        U_norm = torch.nn.functional.normalize(U_k, p=2, dim=1).cpu().numpy()

        # 2. KMeans to get centroids
        kmeans = KMeans(n_clusters=num_clusters, n_init=20, random_state=42)
        kmeans.fit(U_norm)
        centroids = kmeans.cluster_centers_

        # 3. Balanced slots
        base = num_samples // num_clusters
        remainder = num_samples % num_clusters

        slots_per_cluster = [base] * num_clusters
        for cid in range(remainder):
            slots_per_cluster[cid] += 1

        expanded_centroids = []
        slot_cluster_ids = []

        for cid, cnt in enumerate(slots_per_cluster):
            for _ in range(cnt):
                expanded_centroids.append(centroids[cid])
                slot_cluster_ids.append(cid)

        expanded_centroids = np.asarray(expanded_centroids, dtype=np.float32)

        if expanded_centroids.shape[0] != num_samples:
            raise RuntimeError(
                f"Expanded centroid count mismatch: got {expanded_centroids.shape[0]}, expected {num_samples}."
            )

        # 4. Assignment
        dist_matrix = cdist(U_norm, expanded_centroids, metric='euclidean')
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        labels = np.full(num_samples, -1, dtype=int)
        for r, c in zip(row_ind, col_ind):
            labels[r] = slot_cluster_ids[c]

        if np.any(labels < 0):
            missing = np.where(labels < 0)[0].tolist()
            raise RuntimeError(f"Some samples were not assigned in spectral_clustering: {missing}")

        return labels
    
    def predict(self, image, num_clusters, tiles=None, return_tiles=False):
        # Split the image
        if tiles is None:
            tiles, positions = self.split_image_into_tiles(image)
        else:
            positions = [(0, 0)] * len(tiles) 
        
        # Extract features
        features = self.extract_features(tiles)
        
        # Clustering
        labels = self.spectral_clustering(features, num_clusters)
        
        result = {
            'labels': labels,
            'num_clusters': num_clusters,
            'num_pieces': len(tiles),
            'positions': positions,
            'features': features
        }
        
        if return_tiles:
            result['tiles'] = tiles
        
        return result
    
    def group_tiles_by_cluster(self, result):
        tiles = result.get('tiles')
        if tiles is None:
            return None
        
        labels = result['labels']
        clusters = {}
        
        for tile, label in zip(tiles, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(tile)
        
        return clusters
    
    def visualize_result(self, result, save_path=None):
        tiles = result.get('tiles')
        if tiles is None:
            return
        
        clusters = self.group_tiles_by_cluster(result)
        
        # Create grid visualization
        tile_size = self.tile_size
        max_cols = 3
        label_height = 35
        cluster_cols = 6
        
        cluster_ids = sorted(clusters.keys())
        num_clusters = len(cluster_ids)
        cluster_rows = (num_clusters + cluster_cols - 1) // cluster_cols
        
        cluster_blocks = []
        
        for cluster_id in cluster_ids:
            cluster_tiles = clusters[cluster_id]
            
            # Create grid for each cluster
            cols = min(max_cols, len(cluster_tiles))
            rows = (len(cluster_tiles) + cols - 1) // cols
            
            grid = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
            
            for idx, tile in enumerate(cluster_tiles):
                r = idx // cols
                c = idx % cols
                y = r * tile_size
                x = c * tile_size
                
                tile_array = np.array(tile.resize((tile_size, tile_size)))
                grid[y:y+tile_size, x:x+tile_size] = tile_array
            
            # Add label
            h, w = grid.shape[:2]
            label_img = np.zeros((label_height, w, 3), dtype=np.uint8)
            cv2.putText(
                label_img,
                f"Cluster {cluster_id}",
                (5, int(label_height * 0.75)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2
            )
            
            block = np.vstack([label_img, grid])
            cluster_blocks.append(block)
        
        # Create final large grid
        block_heights = [b.shape[0] for b in cluster_blocks]
        block_widths = [b.shape[1] for b in cluster_blocks]
        
        if block_heights:
            max_h = max(block_heights)
            max_w = max(block_widths)
            
            canvas = np.zeros((cluster_rows * max_h, cluster_cols * max_w, 3), dtype=np.uint8)
            
            for idx, block in enumerate(cluster_blocks):
                r = idx // cluster_cols
                c = idx % cluster_cols
                
                h, w = block.shape[:2]
                y = r * max_h
                x = c * max_w
                
                canvas[y:y+h, x:x+w] = block
            
            # Save or display
            if save_path is not None:
                canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, canvas_bgr)
                print(f"Result saved to: {save_path}")
            
            cv2.imshow("DEPP Clustering Result", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @staticmethod
    def _capacity_constrained_assign(cost_matrix, comp_sizes, cluster_capacities):
        """
        Greedy capacity-constrained assignment: allocate num_components "super nodes" to num_clusters clusters,
        ensuring that each cluster's tile total = cluster_capacities[j].
        Assign larger connected components first, then smaller ones, prioritizing the cluster with the smallest cost.
        """
        num_components, num_clusters = cost_matrix.shape
        remaining = list(cluster_capacities)
        labels = np.full(num_components, -1, dtype=int)

        # Assign large connected components first
        order = sorted(range(num_components), key=lambda i: -comp_sizes[i])

        for i in order:
            sz = int(comp_sizes[i])
            feasible = [j for j in range(num_clusters) if remaining[j] >= sz]
            if not feasible:
                # In case of capacity overflow: select the cluster with the largest remaining capacity
                feasible = [max(range(num_clusters), key=lambda j: remaining[j])]
            best = min(feasible, key=lambda j: cost_matrix[i, j])
            labels[i] = best
            remaining[best] -= sz

        return labels

    def constrained_recluster(self, features, tiles, num_clusters,
                               cluster_size=None, must_link=None, bad_assignments=None):
        if features is None or len(features) == 0:
            raise ValueError("features is empty in constrained_recluster.")

        features = np.asarray(features, dtype=np.float32)
        num_samples = features.shape[0]

        if len(tiles) != num_samples:
            raise ValueError(f"Mismatch: tiles={len(tiles)}, features={num_samples}.")
        if num_clusters <= 0:
            raise ValueError(f"num_clusters must be positive, got {num_clusters}.")
        if num_clusters > num_samples:
            num_clusters = num_samples

        # ---- Determine cluster capacity ----
        if cluster_size is not None:
            cluster_capacities = [cluster_size] * num_clusters
        else:
            base = num_samples // num_clusters
            rem  = num_samples % num_clusters
            cluster_capacities = [base + (1 if i < rem else 0) for i in range(num_clusters)]

        # ---- 1. Create connected components (super nodes) from must-link ----
        G = nx.Graph()
        G.add_nodes_from(range(num_samples))
        if must_link:
            G.add_edges_from(must_link)

        components = [sorted(list(c)) for c in nx.connected_components(G)]
        num_components = len(components)

        tile_to_comp = np.zeros(num_samples, dtype=int)
        comp_sizes   = np.zeros(num_components, dtype=int)
        for ci, comp in enumerate(components):
            comp_sizes[ci] = len(comp)
            for ti in comp:
                tile_to_comp[ti] = ci

        # ---- 2. Compute super node features (mean feature of members) ----
        comp_features = np.array(
            [np.mean(features[comp], axis=0) for comp in components],
            dtype=np.float32
        )

        # ---- 3. Centering + SVD dimensionality reduction ----
        F = torch.from_numpy(comp_features).float()
        M = F - F.mean(dim=0, keepdim=True)
        U, S_vals, Vh = torch.linalg.svd(M, full_matrices=False)
        k      = min(num_clusters, U.shape[1])
        U_norm = torch.nn.functional.normalize(U[:, :k], p=2, dim=1).cpu().numpy()

        # ---- 4. KMeans to get centroids ----
        n_init = min(20, max(1, num_components))
        kmeans = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=42)
        kmeans.fit(U_norm)
        centroids = kmeans.cluster_centers_

        # ---- 5. Super node level cost matrix ----
        cost_matrix = cdist(U_norm, centroids, metric='euclidean')  # (num_components, num_clusters)

        # Inject bad_assignments penalty (tile-level to super node-level)
        if bad_assignments:
            penalty = 10.0
            for tile_idx, forbidden_cluster in bad_assignments:
                if not (0 <= tile_idx < num_samples):
                    continue
                if not (0 <= forbidden_cluster < num_clusters):
                    continue
                ci = tile_to_comp[tile_idx]
                cost_matrix[ci, forbidden_cluster] += penalty

        # ---- 6. Capacity constrained assignment (super node level) ----
        comp_labels = self._capacity_constrained_assign(
            cost_matrix, comp_sizes, cluster_capacities
        )

        # ---- 7. Expand super node labels to tile labels ----
        labels = np.full(num_samples, -1, dtype=int)
        for ci, (comp, label) in enumerate(zip(components, comp_labels)):
            for ti in comp:
                labels[ti] = label

        if np.any(labels < 0):
            missing = np.where(labels < 0)[0].tolist()
            raise RuntimeError(f"Some samples not assigned: {missing}")

        return {
            'labels': labels,
            'num_clusters': num_clusters,
            'tiles': tiles
        }