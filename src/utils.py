import os
import ot
import scipy
import torch
import numpy as np
import scanpy as sc
from tqdm import tqdm
from anndata import AnnData
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R


def to_backend_array(x: object, nx: ot.backend.Backend, dtype: object = "float32", use_gpu: bool = True) -> object:
    """
    Convert input `x` into the POT backend format (NumPy, Torch, JAX, etc.),
    while handling device placement and dtype.

    Parameters
    ----------
    x : np.ndarray | torch.Tensor | other backend array
        Input data.
    nx : ot.backend.Backend
        The POT backend (e.g., ot.backend.TorchBackend(), ot.backend.NumpyBackend()).
    dtype : str | torch.dtype
        Desired dtype (default: "float32").
    use_gpu : bool
        If True and backend is Torch, move to GPU if available.
    """
    
    # --- Normalize input into backend ---
    if isinstance(x, np.ndarray):
        x = nx.from_numpy(x)

    elif isinstance(x, torch.Tensor):
        if not isinstance(nx, ot.backend.TorchBackend):
            # Convert Torch tensor → NumPy → target backend
            x = x.detach().cpu().numpy()
            x = nx.from_numpy(x)

    # --- Handle Torch backend specifics ---
    if isinstance(nx, ot.backend.TorchBackend):
        # Convert dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.float32
        x = x.to(dtype)

        # Move to GPU if requested
        device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        x = x.to(device)

    return x


def compute_morphology_cost_matrix(
        sliceA: AnnData,
        sliceB: AnnData,
        nx: ot.backend.Backend,
        use_gpu: bool = False,
        alpha_cell_spatial: float = 0.3,
        verbose: bool = False) -> object:
    """
    Computes cell morphology dissimilarity cost matrix between spots using Fused Gromov-Wasserstein.
    
    For each pair of spots (i, j):
    - Extracts cells assigned to spot i in slice A and spot j in slice B
    - Uses cell morphology features (16D) and cell centroids (2D spatial positions)
    - Computes Fused GW cost that preserves both:
      * Morphological similarity between cells
      * Spatial arrangement of cells within spots
    - Handles spots with no cells using a large penalty value
    
    Args:
        sliceA: First slice with cell_morphology and cell_centroids in uns
        sliceB: Second slice with cell_morphology and cell_centroids in uns
        nx: POT backend
        use_gpu: If True, uses GPU for computation
        alpha_cell_spatial: Weight for cell spatial structure (0=morphology only, 1=spatial only)
        verbose: If True, prints progress
        
    Returns:
        M_morph: Morphology+spatial cost matrix (n_spots_A x n_spots_B)
    """
    print("Computing cell-level Fused Gromov-Wasserstein cost matrix...")
    print(f"  alpha_cell_spatial = {alpha_cell_spatial:.2f} (morphology weight: {1-alpha_cell_spatial:.2f})")
    
    n_spots_A = sliceA.shape[0]
    n_spots_B = sliceB.shape[0]
    
    # Extract morphology and centroid data
    morph_A = sliceA.uns['cell_morphology']
    morph_B = sliceB.uns['cell_morphology']
    centroid_A = sliceA.uns['cell_centroids']
    centroid_B = sliceB.uns['cell_centroids']
    
    # Initialize cost matrix
    M_morph = np.zeros((n_spots_A, n_spots_B))
    
    # Compute maximum possible cost for penalty (will set after computing valid costs)
    valid_costs = []
    
    # First pass: compute costs for spots with cells
    for i in tqdm(range(n_spots_A), desc="Computing cell-level FGW costs"):
        spot_key_A = str(i)
        
        # Get cells for spot i in slice A
        if spot_key_A in morph_A and len(morph_A[spot_key_A]) > 0:
            cells_A = morph_A[spot_key_A]  # shape: (n_cells_i, 16)
            centroids_A = centroid_A[spot_key_A]  # shape: (n_cells_i, 2)
            n_cells_A = cells_A.shape[0]
            a_dist = np.ones(n_cells_A) / n_cells_A  # uniform distribution
        else:
            cells_A = None
            centroids_A = None
            
        for j in range(n_spots_B):
            spot_key_B = str(j)
            
            # Get cells for spot j in slice B
            if spot_key_B in morph_B and len(morph_B[spot_key_B]) > 0:
                cells_B = morph_B[spot_key_B]  # shape: (n_cells_j, 16)
                centroids_B = centroid_B[spot_key_B]  # shape: (n_cells_j, 2)
                n_cells_B = cells_B.shape[0]
                b_dist = np.ones(n_cells_B) / n_cells_B  # uniform distribution
            else:
                cells_B = None
                centroids_B = None
            
            # Compute OT cost if both spots have cells
            if cells_A is not None and cells_B is not None:
                # Convert to backend arrays for GPU acceleration
                cells_A_gpu = to_backend_array(cells_A, nx, use_gpu=use_gpu)
                cells_B_gpu = to_backend_array(cells_B, nx, use_gpu=use_gpu)
                centroids_A_gpu = to_backend_array(centroids_A, nx, use_gpu=use_gpu)
                centroids_B_gpu = to_backend_array(centroids_B, nx, use_gpu=use_gpu)
                a_dist_gpu = to_backend_array(a_dist, nx, use_gpu=use_gpu)
                b_dist_gpu = to_backend_array(b_dist, nx, use_gpu=use_gpu)
                
                # Morphology dissimilarity (like gene expression cost M)
                M_cell_morph = ot.dist(cells_A_gpu, cells_B_gpu, metric='euclidean')
                
                # Intra-spot spatial structure (like D_A, D_B for spots)
                D_cell_A = ot.dist(centroids_A_gpu, centroids_A_gpu, metric='euclidean')
                D_cell_B = ot.dist(centroids_B_gpu, centroids_B_gpu, metric='euclidean')
                
                # Normalize spatial distances to make them comparable
                if n_cells_A > 1:
                    min_dist_A = nx.min(D_cell_A[D_cell_A > 0])
                    if min_dist_A > 0:
                        D_cell_A = D_cell_A / min_dist_A
                if n_cells_B > 1:
                    min_dist_B = nx.min(D_cell_B[D_cell_B > 0])
                    if min_dist_B > 0:
                        D_cell_B = D_cell_B / min_dist_B
                
                # Fused Gromov-Wasserstein for cells within this spot pair
                # This preserves both morphology similarity and spatial arrangement
                try:
                    cost = ot.gromov.fused_gromov_wasserstein2(
                        M_cell_morph,     # Morphology feature cost
                        D_cell_A,         # Spatial structure within spot i
                        D_cell_B,         # Spatial structure within spot j
                        a_dist_gpu, b_dist_gpu,
                        loss_fun='square_loss',
                        alpha=alpha_cell_spatial,  # Balance features vs spatial
                        log=False
                    )
                    # Convert result back to numpy for storage
                    if hasattr(cost, 'item'):
                        cost = cost.item()
                    else:
                        cost = float(nx.to_numpy(cost))
                    M_morph[i, j] = cost
                    valid_costs.append(cost)
                except Exception as e:
                    # Fallback to simple EMD if FGW fails (e.g., single cell)
                    if verbose:
                        print(f"FGW failed for spot pair ({i},{j}), using EMD fallback: {e}")
                    cost = ot.emd2(a_dist_gpu, b_dist_gpu, M_cell_morph)
                    if hasattr(cost, 'item'):
                        cost = cost.item()
                    else:
                        cost = float(nx.to_numpy(cost))
                    M_morph[i, j] = cost
                    valid_costs.append(cost)
            else:
                # Will fill with penalty later
                M_morph[i, j] = -1  # placeholder
    
    # Second pass: set penalty for spots without cells
    if len(valid_costs) > 0:
        max_valid_cost = np.max(valid_costs)
        penalty = max_valid_cost * 10  # 10x maximum valid cost
    else:
        penalty = 1000.0  # fallback if no valid costs
    
    # Replace placeholders with penalty
    M_morph[M_morph < 0] = penalty
    
    if verbose:
        print(f"Cell-level FGW cost matrix computed:")
        print(f"  Shape: {M_morph.shape}")
        print(f"  Valid costs: {len(valid_costs)}/{n_spots_A * n_spots_B}")
        print(f"  Cost range: [{np.min(valid_costs):.4f}, {np.max(valid_costs):.4f}]")
        print(f"  Penalty value: {penalty:.4f}")
    
    # Convert to backend array
    M_morph = to_backend_array(M_morph, nx, use_gpu=use_gpu)
    
    print("Cell-level FGW cost matrix computation complete!")
    return M_morph


def paste_pairwise_align_modified(
        sliceA: AnnData, 
        sliceB: AnnData, 
        alpha: float = 0.1, 
        dissimilarity: str = 'js', 
        sinkhorn: bool = False,
        use_rep: Optional[str] = None,
        lambda_sinkhorn: float = 1, 
        G_init = None, 
        a_distribution = None, 
        b_distribution = None, 
        norm: bool = True, 
        numItermax: int = 10000, 
        backend = ot.backend.NumpyBackend(), 
        use_gpu: bool = False, 
        return_obj: bool = False, 
        verbose: bool = False, 
        cost_mat_path: Optional[str] = None,
        use_morphology: bool = True,
        beta_morphology: float = 1.0,
        alpha_cell_spatial: float = 0.3,
        morphology_cost_path: Optional[str] = None,
        **kwargs) -> Tuple[np.ndarray, Optional[int]]:
        """
        Calculates and returns optimal alignment of two slices using Fused Gromov-Wasserstein transport.
        
        This function serves as the central hub for all backend and GPU management. All computational
        backends and device placement decisions are handled here, with other functions following this
        configuration.
        
        Args:
            sliceA: First slice to align.
            sliceB: Second slice to align.
            alpha: Alignment tuning parameter (0 <= alpha <= 1) for SPOT-level spatial vs gene expression.
            dissimilarity: Expression dissimilarity measure ('kl', 'euclidean', 'js'/'jensenshannon').
            sinkhorn: Must be True for this implementation.
            use_rep: If None, uses slice.X, otherwise uses slice.obsm[use_rep].
            lambda_sinkhorn: Sinkhorn regularization parameter.
            G_init: Initial mapping for FGW-OT (optional).
            a_distribution: Distribution of sliceA spots (optional, defaults to uniform).
            b_distribution: Distribution of sliceB spots (optional, defaults to uniform).
            norm: If True, normalizes spatial distances.
            numItermax: Maximum iterations for FGW-OT.
            backend: POT backend (NumpyBackend or TorchBackend).
            use_gpu: If True, uses GPU (requires TorchBackend).
            return_obj: If True, returns objective function value.
            verbose: If True, enables verbose output.
            cost_mat_path: Path to save/load cost matrix (optional).
            use_morphology: If True, incorporates cell morphology costs (requires cell_morphology in uns).
            beta_morphology: Weight for morphology vs gene expression (0=gene only, 1=morphology only).
            alpha_cell_spatial: Weight for CELL-level spatial structure vs morphology (0=morphology only, 1=spatial only).
            morphology_cost_path: Path to save/load morphology cost matrix (optional).
    
        Returns:
            np.ndarray: Optimal transport matrix.
            Optional[float]: Objective function value (if return_obj=True).
        """

        print("---------------------------------------")
        print('Inside paste_pairwise_align_modified')
        print("---------------------------------------")
                
        # subset for common genes
        common_genes = intersect(sliceA.var.index, sliceB.var.index)
        sliceA = sliceA[:, common_genes]
        sliceB = sliceB[:, common_genes]

        # Use provided backend
        nx = backend  
        
        # Calculate spatial distances
        coordinatesA = sliceA.obsm['spatial'].copy()
        coordinatesA = to_backend_array(coordinatesA, nx, use_gpu=use_gpu)

        coordinatesB = sliceB.obsm['spatial'].copy()
        coordinatesB = to_backend_array(coordinatesB, nx, use_gpu=use_gpu)

        # Calculate spatial distance matrices (device placement handled by to_backend_array)
        D_A = ot.dist(coordinatesA, coordinatesA, metric='euclidean')
        D_A = to_backend_array(D_A, nx, use_gpu=use_gpu)

        D_B = ot.dist(coordinatesB, coordinatesB, metric='euclidean')
        D_B = to_backend_array(D_B, nx, use_gpu=use_gpu)

        # Calculate expression dissimilarity
        A_X = to_dense_array(extract_data_matrix(sliceA,use_rep))
        A_X = to_backend_array(A_X, nx, use_gpu=use_gpu)

        B_X = to_dense_array(extract_data_matrix(sliceB,use_rep))
        B_X = to_backend_array(B_X, nx, use_gpu=use_gpu)

        # Handle cost matrix loading/generation
        M = None
        expected_shape = (sliceA.shape[0], sliceB.shape[0])
        
        if cost_mat_path and os.path.exists(cost_mat_path):
            print("Loading cost matrix from file system...")
            M_loaded = np.load(cost_mat_path)
            if M_loaded.shape == expected_shape:
                M = M_loaded
            else:
                print(f"Loaded cost matrix shape {M_loaded.shape} does not match expected shape {expected_shape}. Regenerating...")
        elif cost_mat_path:
            print("Cost matrix path provided but file does not exist. Generating new cost matrix...")
            
        if M is None:
            # Generate cost matrix based on dissimilarity measure
            dissimilarity_lower = dissimilarity.lower()
            if dissimilarity_lower in ['euclidean', 'euc']:
                M = ot.dist(A_X, B_X)
            elif dissimilarity_lower == 'kl':
                s_A = A_X + 0.01
                s_B = B_X + 0.01
                M = kl_divergence_backend(s_A, s_B, nx)
            elif dissimilarity_lower in ['js', 'jensenshannon']:
                s_A = A_X + 0.01
                s_B = B_X + 0.01
                M = jensenshannon_divergence_backend(s_A, s_B, nx)
            else:
                raise ValueError(f"Unknown dissimilarity measure: {dissimilarity}")
                
            # Save cost matrix if path is provided
            if cost_mat_path:
                np.save(cost_mat_path, nx.to_numpy(M))
                
        M = to_backend_array(M, nx, use_gpu=use_gpu)
        
        # Compute and incorporate morphology costs if requested
        if use_morphology:
            print("Incorporating cell morphology costs...")
            
            # Check if morphology and centroid data exists
            if 'cell_morphology' not in sliceA.uns or 'cell_morphology' not in sliceB.uns:
                raise ValueError("use_morphology=True but cell_morphology not found in slice.uns")
            if 'cell_centroids' not in sliceA.uns or 'cell_centroids' not in sliceB.uns:
                raise ValueError("use_morphology=True but cell_centroids not found in slice.uns")
            
            # Try to load morphology cost matrix
            M_morph = None
            if morphology_cost_path and os.path.exists(morphology_cost_path):
                print("Loading morphology cost matrix from file...")
                M_morph_loaded = np.load(morphology_cost_path)
                if M_morph_loaded.shape == expected_shape:
                    M_morph = M_morph_loaded
                else:
                    print(f"Loaded morphology cost shape {M_morph_loaded.shape} != expected {expected_shape}. Regenerating...")
            
            # Compute if not loaded
            if M_morph is None:
                M_morph = compute_morphology_cost_matrix(sliceA, sliceB, nx, use_gpu=use_gpu, 
                                                         alpha_cell_spatial=alpha_cell_spatial, 
                                                         verbose=verbose)
                M_morph = nx.to_numpy(M_morph)
                
                # Save if path provided
                if morphology_cost_path:
                    np.save(morphology_cost_path, M_morph)
                    print(f"Saved morphology cost matrix to {morphology_cost_path}")
            
            M_morph = to_backend_array(M_morph, nx, use_gpu=use_gpu)
            
            # Normalize both cost matrices to [0, 1] range for fair combination
            M_gene = M  # rename for clarity
            M_gene_min = nx.min(M_gene)
            M_gene_max = nx.max(M_gene)
            M_gene_normalized = (M_gene - M_gene_min) / (M_gene_max - M_gene_min + 1e-10)
            
            M_morph_min = nx.min(M_morph)
            M_morph_max = nx.max(M_morph)
            M_morph_normalized = (M_morph - M_morph_min) / (M_morph_max - M_morph_min + 1e-10)
            
            # Combine: M_combined = (1-beta) * M_gene + beta * M_morph
            M = (1 - beta_morphology) * M_gene_normalized + beta_morphology * M_morph_normalized
            
            if verbose:
                print(f"Combined cost matrix:")
                print(f"  Gene expression weight: {1 - beta_morphology:.2f}")
                print(f"  Cell morphology+spatial weight: {beta_morphology:.2f}")
                print(f"    └─ Within cell matching: morphology weight = {1 - alpha_cell_spatial:.2f}, spatial weight = {alpha_cell_spatial:.2f}")
                print(f"  Gene cost range (normalized): [0, 1]")
                print(f"  Morphology cost range (normalized): [0, 1]")
        
        # Initialize probability distributions
        if a_distribution is None:
            a = np.ones((sliceA.shape[0],)) / sliceA.shape[0]
        a = to_backend_array(a, nx, use_gpu=use_gpu)
            
        if b_distribution is None:
            b = np.ones((sliceB.shape[0],)) / sliceB.shape[0]
        b = to_backend_array(b, nx, use_gpu=use_gpu)

        # Normalize spatial distances if requested
        if norm:
            D_A /= nx.min(D_A[D_A > 0])
            D_B /= nx.min(D_B[D_B > 0])
        
        # Handle initial mapping
        if G_init is not None:
            G_init = to_backend_array(G_init, nx, use_gpu=use_gpu)
            expected_g_shape = (sliceA.shape[0], sliceB.shape[0])
            if G_init.shape != expected_g_shape:
                print(f"Warning: G_init has shape {G_init.shape} but expected shape {expected_g_shape}. Using default initialization.")
                G_init = None
        
        # Ensure sinkhorn mode is enabled
        if not sinkhorn:
            raise ValueError("This implementation requires sinkhorn=True")

        # Run optimal transport optimization
        pi, logw = my_fused_gromov_wasserstein_gcg(
            M, D_A, D_B, a, b, nx, 
            lambda_sinkhorn=lambda_sinkhorn, 
            G_init=G_init, 
            loss_fun='square_loss', 
            alpha=alpha, 
            log=True, 
            numItermax=numItermax, 
            verbose=verbose, 
            **kwargs
        )
        
        # Convert results back to numpy and clean up GPU memory if needed
        pi = nx.to_numpy(pi)
        obj = nx.to_numpy(logw['fgw_dist'])
        
        # Clear GPU cache to free memory after large computation
        if isinstance(nx, ot.backend.TorchBackend):
            torch.cuda.empty_cache()

        return (pi, obj) if return_obj else pi


def my_fused_gromov_wasserstein_gcg(M, C1, C2, p, q, nx, lambda_sinkhorn=1, G_init = None, loss_fun='square_loss', alpha=0.5, log=False, numItermax=200, **kwargs):
        """
        Adapted fused_gromov_wasserstein with the added capability of defining a G_init (inital mapping).
        All tensors and backend are provided by the calling function.
        
        For more info, see: https://pythonot.github.io/gen_modules/ot.gromov.html
        """
        print("---------------------------------------")
        print("Inside my_fused_gromov_wasserstein_gcg")
        print(f"M shape: {M.shape}")
        print(f"p shape: {p.shape}")
        print(f"q shape: {q.shape}")
        print(f"C1 shape: {C1.shape}")
        print(f"C2 shape: {C2.shape}")
        print("---------------------------------------")
        
        # Print backend information for all variables
        print(f"Backend nx: {type(nx)}")
        print(f"M backend: {type(M)}" + (f", device: {M.device}" if hasattr(M, 'device') else ""))
        print(f"C1 backend: {type(C1)}" + (f", device: {C1.device}" if hasattr(C1, 'device') else ""))
        print(f"C2 backend: {type(C2)}" + (f", device: {C2.device}" if hasattr(C2, 'device') else ""))
        print(f"p backend: {type(p)}" + (f", device: {p.device}" if hasattr(p, 'device') else ""))
        print(f"q backend: {type(q)}" + (f", device: {q.device}" if hasattr(q, 'device') else ""))
        print("---------------------------------------")

        # Validate matrix shapes
        n_a, n_b = len(p), len(q)
        if M.shape != (n_a, n_b):
            raise ValueError(f"Cost matrix M has shape {M.shape} but expected shape ({n_a}, {n_b})")
        if C1.shape != (n_a, n_a):
            raise ValueError(f"Distance matrix C1 has shape {C1.shape} but expected shape ({n_a}, {n_a})")
        if C2.shape != (n_b, n_b):
            raise ValueError(f"Distance matrix C2 has shape {C2.shape} but expected shape ({n_b}, {n_b})")

        constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)

        if G_init is None:
            G0 = p[:, None] * q[None, :]
            print(f"G0 initialized with shape: {G0.shape}")
        else:
            G0 = (1/nx.sum(G_init)) * G_init
            print(f"G0 from G_init with shape: {G0.shape}")
        
        # Print backend information for computed variables
        print(f"constC backend: {type(constC)}" + (f", device: {constC.device}" if hasattr(constC, 'device') else ""))
        print(f"hC1 backend: {type(hC1)}" + (f", device: {hC1.device}" if hasattr(hC1, 'device') else ""))
        print(f"hC2 backend: {type(hC2)}" + (f", device: {hC2.device}" if hasattr(hC2, 'device') else ""))
        print(f"G0 backend: {type(G0)}" + (f", device: {G0.device}" if hasattr(G0, 'device') else ""))
        print("---------------------------------------")

        def f(G):
            return ot.gromov.gwloss(constC, hC1, hC2, G)

        def df(G):
            return ot.gromov.gwggrad(constC, hC1, hC2, G)

        if log:
            print('log true')
            res, log = ot.optim.gcg(p, q, M, lambda_sinkhorn, alpha, f, df, G0, log=True, **kwargs)
            fgw_dist = log['loss'][-1]
            log['fgw_dist'] = fgw_dist
            return res, log

        else:
            print('log false')
            pi = ot.optim.gcg(p, q, M, lambda_sinkhorn, alpha, f, df, G0, log=False, **kwargs)
            return pi, -1


def filter_for_common_genes(
    slices: List[AnnData]) -> None:
    """
    Filters for the intersection of genes between all slices.

    Args:
        slices: List of slices.
    """
    assert len(slices) > 0, "Cannot have empty list."

    common_genes = slices[0].var.index
    for s in slices:
        common_genes = intersect(common_genes, s.var.index)
    for i in range(len(slices)):
        slices[i] = slices[i][:, common_genes]
    print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')


def kl_divergence_backend(X, Y, nx):
    """
    Returns pairwise KL divergence using the provided backend.

    Args:
        X: backend tensor with dim (n_samples by n_features)
        Y: backend tensor with dim (m_samples by n_features)
        nx: POT backend to use for computation

    Returns:
        D: backend tensor with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X,log_Y.T)
    return D

def kl_divergence_corresponding_backend(X, Y, nx):
    """
    Returns corresponding KL divergence using the provided backend.

    Args:
        X: backend tensor with dim (n_samples by n_features)
        Y: backend tensor with dim (n_samples by n_features)
        nx: POT backend to use for computation

    Returns:
        D: backend tensor with dim (n_samples by 1). Corresponding KL divergence.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i',X,log_X)
    X_log_X = nx.reshape(X_log_X,(1,X_log_X.shape[0]))

    X_log_Y = nx.einsum('ij,ij->i',X,log_Y)
    X_log_Y = nx.reshape(X_log_Y,(1,X_log_Y.shape[0]))
    D = X_log_X.T - X_log_Y.T
    return D

def jensenshannon_distance_1_vs_many_backend(X, Y, nx):
    """
    Returns pairwise Jensen-Shannon distance using the provided backend.

    Args:
        X: backend tensor with dim (1 by n_features)
        Y: backend tensor with dim (m_samples by n_features)
        nx: POT backend to use for computation

    Returns:
        js_dist: backend tensor with Jensen-Shannon distances.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    assert X.shape[0] == 1

    X = nx.concatenate([X] * Y.shape[0], axis=0)
    X = X/nx.sum(X,axis=1, keepdims=True)
    Y = Y/nx.sum(Y,axis=1, keepdims=True)
    M = (X + Y) / 2.0
    kl_X_M = kl_divergence_corresponding_backend(X, M, nx)
    kl_Y_M = kl_divergence_corresponding_backend(Y, M, nx)
    js_dist = nx.sqrt((kl_X_M + kl_Y_M) / 2.0).T[0]
    return js_dist


def jensenshannon_divergence_backend(X, Y, nx):
    """
    Returns pairwise Jensen-Shannon divergence using the provided backend.

    Args:
        X: backend tensor with dim (n_samples by n_features)
        Y: backend tensor with dim (m_samples by n_features)
        nx: POT backend to use for computation

    Returns:
        D: backend tensor with dim (n_samples by m_samples). Pairwise JS divergence matrix.
    """
    print("Calculating cost matrix")

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    print(nx.unique(nx.isnan(X)))
    print(nx.unique(nx.isnan(Y)))
        
    X = X/nx.sum(X, axis=1, keepdims=True)
    Y = Y/nx.sum(Y, axis=1, keepdims=True)

    n = X.shape[0]
    m = Y.shape[0]
    
    js_dist = nx.zeros((n, m))

    for i in tqdm(range(n)):
        js_dist[i, :] = jensenshannon_distance_1_vs_many_backend(X[i:i+1], Y, nx)
        
    print("Finished calculating cost matrix")
    print(nx.unique(nx.isnan(js_dist)))

    return js_dist


def intersect(lst1, lst2):
    """
    Gets and returns intersection of two lists.

    Args:
        lst1: List
        lst2: List

    Returns:
        lst3: List of common elements.
    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

## Convert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X,scipy.sparse.csr.spmatrix) else np.array(X)


## Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep]


def rotate(v, angle_deg, center=(0, 0)):
    '''
    v         : numpy array of n 2D points. Shape: (n x 2)
    angle_deg : rotation angle in degrees
    center    : all the points of v will be rotated with respect to center by angle_deg
    '''
    v[:, 0] = v[:, 0] - center[0]
    v[:, 1] = v[:, 1] - center[1]
    rot_mat_2D = R.from_euler('z', angle_deg, degrees=True).as_matrix()[:2, :2]
    v = (rot_mat_2D @ v.T).T
    v[:, 0] = v[:, 0] + center[0]
    v[:, 1] = v[:, 1] + center[1]
    return v


def compute_null_distribution(pi, cost_mat, scheme):
    if scheme == 'all_edges':
        non_zero_idxs_pi = np.nonzero(pi.flatten())[0]
        distances = cost_mat.flatten()[non_zero_idxs_pi]
        weights = pi.flatten()[non_zero_idxs_pi]
    elif scheme == 'left':
        score_mat = pi * cost_mat
        distances = np.sum(score_mat, axis=1) / (1 / pi.shape[0]) * 100
        # print('left', distances.min(), distances.max())
        weights = [1] * len(distances)
    elif scheme == 'right':
        score_mat = pi * cost_mat
        distances = np.sum(score_mat, axis=0) / (1 / pi.shape[1]) * 100
        # print('right', distances.min(), distances.max())
        weights = [1] * len(distances)
    else:
        print("Please set a valid scheme! \n"
              "a) all_edges\n"
              "b) left\n"
              "c) right\n"
              "(at compute_null_distribution function in utils.py)")
        
    return distances, weights


def QC(adata):
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    sc.pp.scale(adata, max_value=10)


def scale_coords(adata, key_name):
    adata.obsm[key_name] = adata.obsm[key_name].astype('float')
    x = adata.obsm[key_name][:, 0]
    y = adata.obsm[key_name][:, 1]
    adata.obsm[key_name][:, 0] = x / x.max()
    adata.obsm[key_name][:, 1] = y / y.max()
