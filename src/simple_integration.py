"""
Simple integration: Assign cells to nearest spot, store in h5ad.
"""

import os
import anndata
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Paths
data_dir = r'c:\Users\afzal\OneDrive\Desktop\HOT\Data\John_Roger'
h5ad_path = os.path.join(data_dir, 'pod.h5ad')
csv_path = os.path.join(data_dir, 'pod.csv')
output_path = os.path.join(data_dir, 'pod_m.h5ad')

# Load data
print("Loading data...")
adata = anndata.read_h5ad(h5ad_path)
morph_df = pd.read_csv(csv_path)

# Get coordinates
spot_coords = adata.obsm['spatial']
cell_coords = morph_df[['centroid_x', 'centroid_y']].values

# Find nearest spot for each cell
print("Assigning cells to spots...")
tree = cKDTree(spot_coords)
distances, spot_indices = tree.query(cell_coords)

# Select morphology features (all except Object ID and coordinates)
exclude_cols = ['nucleus_id', 'centroid_x', 'centroid_y']
features = [col for col in morph_df.columns if col not in exclude_cols]

morph_data = morph_df[features].values

# Group cells by spot using numeric indices
print("Grouping by spot...")
spot_cells = {}
spot_centroids = {}
for cell_idx, spot_idx in enumerate(spot_indices):
    if spot_idx not in spot_cells:
        spot_cells[spot_idx] = []
        spot_centroids[spot_idx] = []
    spot_cells[spot_idx].append(morph_data[cell_idx])
    spot_centroids[spot_idx].append(cell_coords[cell_idx])

# Convert to arrays and string keys (h5ad requirement)
print("Storing in h5ad...")
spot_cells_str = {}
spot_centroids_str = {}
for spot_idx in spot_cells:
    spot_cells_str[str(spot_idx)] = np.array(spot_cells[spot_idx])
    spot_centroids_str[str(spot_idx)] = np.array(spot_centroids[spot_idx])

# Store in uns
adata.uns['cell_morphology'] = spot_cells_str
adata.uns['cell_centroids'] = spot_centroids_str
adata.uns['morphology_features'] = features

# Add cell count
adata.obs['n_cells'] = [len(spot_cells_str.get(str(i), [])) for i in range(len(adata))]

# Save
print(f"Saving to {output_path}...")
adata.write_h5ad(output_path)

print("\n✓ Done!")
print(f"Spots: {len(adata)}")
print(f"Cells: {len(morph_df)}")
print(f"Spots with cells: {len(spot_cells_str)}")
print(f"\nAccess morphology: adata.uns['cell_morphology'][str(spot_index)]")
print(f"Access centroids:  adata.uns['cell_centroids'][str(spot_index)]")
print(f"Example: adata.uns['cell_morphology']['100']  # shape (n_cells, 16)")
print(f"Example: adata.uns['cell_centroids']['100']   # shape (n_cells, 2)")
