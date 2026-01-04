import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import itertools

class ShiftGraph:
    def __init__(self, adata, n_neighbors=15, n_pcs=30, rep='X_pca'):
        """
        Initializes the graph structure for the entire dataset once.
        
        Args:
            adata: The AnnData object (containing all potential train/test data).
            n_neighbors: Number of neighbors for the k-NN graph.
            n_pcs: Number of PCs to use if PCA needs to be calculated.
            rep: Representation to use (e.g., 'X_pca', 'X').
        """
        self.adata = adata
        self.n_neighbors = n_neighbors
        
        # 1. Ensure PCA exists
        if rep == 'X_pca' and 'X_pca' not in adata.obsm:
            print("Computing PCA...")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
            sc.tl.pca(adata, n_comps=n_pcs)
            
        # 2. Build the Global Graph (The expensive part)
        print(f"Building global k-NN graph (k={n_neighbors})...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=rep, key_added='shift_graph')
        
        # 3. Store the sparse connectivity matrix
        # (This matrix contains the 'edges' of the graph)
        self.connectivities = adata.obsp['shift_graph_connectivities']
        
        # Pre-calculate row indices for fast lookup
        # rows, cols = nonzero indices of the adjacency matrix
        self.edge_rows, self.edge_cols = self.connectivities.nonzero()
        self.n_total_edges = len(self.edge_rows)
        
        print(f"Graph Ready: {adata.n_obs} cells, {self.n_total_edges} edges.")

    def assess_shift(self, split_col, train_vals, test_vals):
        """
        Calculates the Cross-Match Shift Score for a specific Train/Test split.
        
        Args:
            split_col: Column name in adata.obs (e.g., 'City').
            train_vals: List of values for Train (e.g., ['Beijing', 'Shanghai']).
            test_vals: List of values for Test (e.g., ['Wuhan']).
            
        Returns:
            score: The 'Cross-Match Ratio'. 
                   ~0.5 means Mixed (Easy/No Shift).
                   ~0.0 means Separated (Severe Shift).
        """
        # Create boolean masks for this specific split
        # We use .values for faster numpy access
        obs_col = self.adata.obs[split_col].values
        
        is_train = np.isin(obs_col, train_vals)
        is_test = np.isin(obs_col, test_vals)
        
        # We only care about edges that involve cells from THIS split
        # Map global indices to boolean arrays
        
        # Get the status of the "source" node and "target" node for every edge in the graph
        # edge_rows = index of source node
        # edge_cols = index of target node
        
        train_source = is_train[self.edge_rows]
        train_target = is_train[self.edge_cols]
        
        test_source = is_test[self.edge_cols] # Undirected, so check both ways conceptually
        test_target = is_test[self.edge_rows]
        
        # A "Cross-Match" is an edge between a Train cell and a Test cell
        # Logic: (Source is Train AND Target is Test) OR (Source is Test AND Target is Train)
        is_cross = (train_source & test_source) | (test_target & train_target)
        
        # We must normalize by the total valid edges in this subgraph
        # Valid edge = connects (Train or Test) to (Train or Test)
        # We ignore edges connecting 'Train' to 'Unused City'
        is_valid_edge = (
            ((train_source | test_target) & (train_target | test_source))
        )
        
        n_cross = np.sum(is_cross)
        n_valid = np.sum(is_valid_edge)
        
        if n_valid == 0:
            return 0.0 # Should not happen unless empty split
            
        ratio = n_cross / n_valid
        return ratio

    def find_most_severe_split(self, split_col, strategy='one_vs_rest'):
        """
        Automatically finds the hardest Train/Test combination.
        
        Args:
            split_col: Column to iterate over (e.g., 'City').
            strategy: 'one_vs_rest' (Train on all-1, Test on 1) OR 'pairwise' (Train A, Test B).
            
        Returns:
            DataFrame of all splits ranked by severity (lowest score = hardest).
        """
        unique_vals = self.adata.obs[split_col].unique()
        results = []
        
        print(f"Searching for worst shift in '{split_col}' using '{strategy}' strategy...")
        
        if strategy == 'one_vs_rest':
            # Try setting each value as the Test Set (Leave-One-Out)
            for test_val in unique_vals:
                train_vals = [v for v in unique_vals if v != test_val]
                
                score = self.assess_shift(split_col, train_vals, [test_val])
                
                results.append({
                    'Train': "All Others",
                    'Test': test_val,
                    'Shift_Score': score,
                    'Severity': 'Severe' if score < 0.1 else 'Moderate' if score < 0.3 else 'Low'
                })
                
        elif strategy == 'pairwise':
            # Compare every pair (e.g., Beijing vs Wuhan)
            for v1, v2 in itertools.combinations(unique_vals, 2):
                score = self.assess_shift(split_col, [v1], [v2])
                results.append({
                    'Train': v1,
                    'Test': v2,
                    'Shift_Score': score,
                    'Severity': 'Severe' if score < 0.1 else 'Moderate' if score < 0.3 else 'Low'
                })
        
        df = pd.DataFrame(results).sort_values('Shift_Score')
        return df

    def assess_class_conditional_shift(self, split_col, train_vals, test_vals, class_col='majorType'):
        """
        Calculates shift scores for EACH cell type individually.
        Useful to find which specific cell type is causing the batch effect.
        """
        # 1. Identify common classes in this split
        obs = self.adata.obs
        is_train = obs[split_col].isin(train_vals)
        is_test = obs[split_col].isin(test_vals)
        
        train_classes = set(obs.loc[is_train, class_col].unique())
        test_classes = set(obs.loc[is_test, class_col].unique())
        common_classes = sorted(list(train_classes & test_classes))
        
        results = []
        
        print(f"Analyzing shift per class ({len(common_classes)} shared classes)...")
        
        for cls in common_classes:
            # Create a mask that isolates THIS class in THIS split
            # Logic: (Row is Train OR Test) AND (Row is Class X)
            cls_mask = (obs[class_col] == cls)
            
            # We need to pass the *subset* indices to the calculation
            # But our graph is global. 
            # Trick: We can just use the assess_shift logic but conceptually 
            # treat "Train" as "Train + Class X" and "Test" as "Test + Class X"
            
            # Actually, reusing the graph for subsets is tricky because the 
            # 'global' edges might connect Class A to Class B.
            # For a pure class-conditional test, strict rigor requires a subgraph.
            # However, checking the *global* graph for same-class connections is a valid proxy.
            
            # Let's do it manually on the edges to be safe:
            # 1. Get indices of all cells in this class
            cls_indices = np.where(cls_mask.values)[0]
            
            # 2. Filter the pre-computed edge list to keep only edges 
            # where BOTH source and target are in this class
            # (This creates a virtual subgraph)
            mask_source = np.isin(self.edge_rows, cls_indices)
            mask_target = np.isin(self.edge_cols, cls_indices)
            valid_edge_mask = mask_source & mask_target
            
            if np.sum(valid_edge_mask) < 10:
                continue # Skip if too few edges
                
            # 3. Among these "Class X" edges, how many cross Train/Test?
            # We reuse the boolean vectors from the full dataset
            sub_rows = self.edge_rows[valid_edge_mask]
            sub_cols = self.edge_cols[valid_edge_mask]
            
            is_train_vals = is_train.values
            is_test_vals = is_test.values
            
            # Source is Train, Target is Test (or vice versa)
            is_cross = (is_train_vals[sub_rows] & is_test_vals[sub_cols]) | \
                       (is_test_vals[sub_rows] & is_train_vals[sub_cols])
            
            ratio = np.sum(is_cross) / len(sub_rows)
            
            results.append({
                'Class': cls,
                'Shift_Score': ratio,
                'Status': 'Severe' if ratio < 0.1 else 'Moderate'
            })
            
        return pd.DataFrame(results).sort_values('Shift_Score')

    def get_p_value(self, split_col, train_vals, test_vals, n_perms=100):
        """
        Runs a permutation test to assess significance.
        Shuffles the 'Train/Test' labels n_perms times.
        """
        real_score = self.assess_shift(split_col, train_vals, test_vals)
        
        # Identify the pool of cells involved in this split
        obs = self.adata.obs
        mask_involved = obs[split_col].isin(train_vals + test_vals).values
        involved_indices = np.where(mask_involved)[0]
        
        perm_scores = []
        
        # Pre-calculate counts to shuffle efficiently
        n_train = np.sum(obs[split_col].isin(train_vals))
        
        print(f"Calculating p-value ({n_perms} permutations)...")
        
        for i in range(n_perms):
            # Create a random shuffle of labels JUST for the involved cells
            # We assign n_train cells to be 'Fake Train' and the rest 'Fake Test'
            shuffled_indices = np.random.permutation(involved_indices)
            fake_train_idx = set(shuffled_indices[:n_train])
            
            # Calculate score on the graph using these fake labels
            # (We do this manually on edges to avoid editing adata.obs)
            
            # Filter edges to only those within the involved set
            mask_edges = np.isin(self.edge_rows, involved_indices) & \
                         np.isin(self.edge_cols, involved_indices)
            
            sub_rows = self.edge_rows[mask_edges]
            sub_cols = self.edge_cols[mask_edges]
            
            # Check if edge connects FakeTrain to FakeTest
            # Fast lookup: is the row index in the set?
            # Note: Set lookup in Python is O(1), but for array it's slow. 
            # For speed in Python loop, we use a boolean array.
            is_fake_train = np.zeros(self.adata.n_obs, dtype=bool)
            is_fake_train[list(fake_train_idx)] = True
            
            source_train = is_fake_train[sub_rows]
            target_train = is_fake_train[sub_cols]
            
            # Cross match: One is True (Train), One is False (Test)
            is_cross = source_train != target_train
            
            perm_score = np.sum(is_cross) / len(sub_rows)
            perm_scores.append(perm_score)
            
        # P-value = Fraction of random scores that are WORSE (lower) than real score
        # Since lower score = more separation, we check if random < real.
        # Actually, random mixing is usually HIGH (~0.5). Real shift is LOW (~0.1).
        # So we check: How often is random score <= real score?
        # If real is 0.1 and random is always 0.5, p-value is 0/100 = 0.0.
        n_more_extreme = np.sum(np.array(perm_scores) <= real_score)
        p_val = (n_more_extreme + 1) / (n_perms + 1) # +1 for pseudocount
        
        return real_score, p_val

    def visualize_split(self, split_col, train_vals, test_vals, embed='X_pca'):
        """
        Plots the embedding with Train/Test colored.
        """
        obs = self.adata.obs.copy()
        
        # Create a temporary column for plotting
        obs['Split_Viz'] = 'Other'
        obs.loc[obs[split_col].isin(train_vals), 'Split_Viz'] = 'Train'
        obs.loc[obs[split_col].isin(test_vals), 'Split_Viz'] = 'Test'
        
        # Filter out 'Other' for cleaner plot
        plot_mask = obs['Split_Viz'] != 'Other'
        
        # Get coordinates
        if embed in self.adata.obsm:
            coords = self.adata.obsm[embed]
        else:
            print(f"Embedding {embed} not found.")
            return

        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        
        # Plot Train
        idx_train = np.where((obs['Split_Viz'] == 'Train') & plot_mask)[0]
        plt.scatter(coords[idx_train, 0], coords[idx_train, 1], 
                   c='blue', s=1, alpha=0.5, label='Train')
        
        # Plot Test
        idx_test = np.where((obs['Split_Viz'] == 'Test') & plot_mask)[0]
        plt.scatter(coords[idx_test, 0], coords[idx_test, 1], 
                   c='red', s=1, alpha=0.5, label='Test')
        
        plt.legend()
        plt.title(f"Split Visualization: {embed}")
        plt.show()