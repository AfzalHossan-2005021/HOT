import os
import ot
import math
import scipy
import torch
import random
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics
from DataLoader import DataLoader
from utils import compute_null_distribution, scale_coords, QC, paste_pairwise_align_modified

class AnalyzeOutput:
    def __init__(self, config):
        self.config = config
        self.dataset = config['dataset']
        self.sample_left = config['sample_left']
        self.sample_right = config['sample_right']
        self.alpha = config['alpha']
        self.numIterMaxEmd = config['numIterMaxEmd']
        self.use_gpu = config['use_gpu']
        self.results_path = config['results_path']
        self.pi = config['pi']
        self.config_file_name = os.path.basename(self.config['config_path'])
        self.dissimilarity = config['dissimilarity']
        self.lambda_sinkhorn = config['lambda_sinkhorn']
        self.beta_morphology = config['beta_morphology']
        self.alpha_cell_spatial = config.get('alpha_cell_spatial', 0.3)  # Default to 0.3 for backward compatibility
        self.sinkhorn = config['sinkhorn']
        self.numInnerIterMax = config['numInnerIterMax']
        self.grid_search = config['grid_search']

        if config['adata_left_path'] != 'None':
            self.adata_left = sc.read(config['adata_left_path'])
            self.adata_right = sc.read(config['adata_right_path'])
        else:
            data_loader = DataLoader(config)
            dataset_map = data_loader.read_data(self.dataset)

            self.adata_left = dataset_map[self.sample_left]
            self.adata_right = dataset_map[self.sample_right]
                
        config['cost_mat'] = np.load(config['cost_mat_path'])
        config['morphology_cost'] = np.load(config['morphology_cost_path'])
        self.cost_mat_gene = config['cost_mat']
        self.cost_mat_morphology = config['morphology_cost']
        
        # Combine cost matrices using beta_morphology (same as alignment)
        self.cost_mat = self._combine_cost_matrices(
            self.cost_mat_gene, 
            self.cost_mat_morphology, 
            self.beta_morphology
        )

        scale_coords(self.adata_left, key_name='spatial')
        scale_coords(self.adata_right, key_name='spatial')

        if config['QC']:
            QC(self.adata_left)
            QC(self.adata_right)

        self.fig_hist_rs, self.ax_hist_rs = plt.subplots()
        self.ax_hist_rs.set_xlabel('Remodeling score')
        self.ax_hist_rs.set_ylabel('Count')

    def _combine_cost_matrices(self, M_gene, M_morph, beta_morphology):
        """Combines gene expression and morphology cost matrices using beta_morphology weight.
        
        Uses same normalization and combination as paste_pairwise_align_modified.
        """
        # Normalize both cost matrices to [0, 1] range for fair combination
        M_gene_min = np.min(M_gene)
        M_gene_max = np.max(M_gene)
        M_gene_normalized = (M_gene - M_gene_min) / (M_gene_max - M_gene_min + 1e-10)
        
        M_morph_min = np.min(M_morph)
        M_morph_max = np.max(M_morph)
        M_morph_normalized = (M_morph - M_morph_min) / (M_morph_max - M_morph_min + 1e-10)
        
        # Combine: M_combined = (1-beta) * M_gene + beta * M_morph
        M_combined = (1 - beta_morphology) * M_gene_normalized + beta_morphology * M_morph_normalized
        
        print(f"Combined cost matrix using beta_morphology={beta_morphology:.2f}")
        print(f"  Gene expression weight: {1 - beta_morphology:.2f}")
        print(f"  Cell morphology weight: {beta_morphology:.2f}")
        
        return M_combined

    def visualize_goodness_of_mapping(self, slice_pos='right', invert_x=False):
        if slice_pos == 'left': 
            adata = self.adata_left
            pi = self.pi
            cost_mat = self.cost_mat
            sample_name = self.sample_left
        else:
            adata = self.adata_right
            pi = self.pi.T
            cost_mat = self.cost_mat.T
            sample_name = self.sample_right

        score_mat = pi * cost_mat

        adata.obs['pathological_score'] = np.sum(score_mat, axis=1, dtype=np.float64) / (1 / adata.n_obs) * 100
        adata.obs['pathological_score'].to_csv(f'{self.results_path}/{self.dataset}/{self.config_file_name}/pathological_scores.csv')

        bins = 100
        plt.figure(figsize=(9, 9))
        plt.hist(adata.obs['pathological_score'].values, bins=bins)
        os.makedirs(f'{self.results_path}/{self.dataset}/{self.config_file_name}/Histograms/', exist_ok=True)
        plt.savefig(f'{self.results_path}/{self.dataset}/{self.config_file_name}/Histograms/{slice_pos}_pathological_score.jpg',format='jpg',dpi=350,bbox_inches='tight',pad_inches=0)
        plt.close()

        f, ax = plt.subplots()
        plt.figure(figsize=(9, 9))
        ax.axis('off')
        points = ax.scatter(-adata.obsm['spatial'][:, 0], -adata.obsm['spatial'][:, 1], s=10, c=adata.obs['pathological_score'].values, cmap='plasma_r')
        if invert_x:
            f.gca().invert_xaxis()
        f.colorbar(points)
        config_file_name = os.path.basename(self.config['config_path'])
        os.makedirs(f'{self.results_path}/{self.dataset}/{config_file_name}/Pathology_score/', exist_ok=True)
        f.savefig(f'{self.results_path}/{self.dataset}/{config_file_name}/Pathology_score/{sample_name}_pathology_score.jpg',format='jpg',dpi=350,bbox_inches='tight',pad_inches=0)
        f.savefig(f'{self.results_path}/{self.dataset}/{config_file_name}/Pathology_score/{sample_name}_pathology_score.eps',format='eps',dpi=350,bbox_inches='tight',pad_inches=0)
        f.savefig(f'{self.results_path}/{self.dataset}/{config_file_name}/Pathology_score/{sample_name}_pathology_score.svg',format='svg',dpi=350,bbox_inches='tight',pad_inches=0)
        plt.close()

        # Approach 2: Hexbin plot for density visualization
        f2, ax2 = plt.subplots(figsize=(15, 15))
        ax2.axis('off')
        x_coords = -adata.obsm['spatial'][:, 0]
        y_coords = -adata.obsm['spatial'][:, 1]
        hexbin = ax2.hexbin(
            x_coords, y_coords,
            extent=[x_coords.min(), x_coords.max(),
                    y_coords.min(), y_coords.max()],
            C=adata.obs['pathological_score'].values,
            gridsize=75, cmap='plasma_r', reduce_C_function=np.mean
        )
        if invert_x:
            f2.gca().invert_xaxis()
        f2.colorbar(hexbin, label='Mean Pathological Score')
        ax2.set_title(f'{sample_name} - Pathological Score (Hexbin Density)')
        f2.savefig(f'{self.results_path}/{self.dataset}/{config_file_name}/Pathology_score/{sample_name}_pathology_score_hexbin.jpg',format='jpg',dpi=350,bbox_inches='tight',pad_inches=0)
        f2.savefig(f'{self.results_path}/{self.dataset}/{config_file_name}/Pathology_score/{sample_name}_pathology_score_hexbin.eps',format='eps',dpi=350,bbox_inches='tight',pad_inches=0)
        f2.savefig(f'{self.results_path}/{self.dataset}/{config_file_name}/Pathology_score/{sample_name}_pathology_score_hexbin.svg',format='svg',dpi=350,bbox_inches='tight',pad_inches=0)
        plt.close()


    def divide_into_2_regions_wrt_goodness_score_and_find_DEG(self):
        adata_healthy_left = self.adata_left.copy()

        if self.config['adata_healthy_right_path'] != 'None':
            adata_healthy_right = sc.read(self.config['adata_healthy_right_path'])
        else:
            adata_healthy_right = None

        right_threshold = self.get_goodness_threshold_from_null_distribution(adata_healthy_left, adata_healthy_right)

        print('Thresholds:', right_threshold)
        df_right_threshold = pd.DataFrame({'right_threshold': [right_threshold]})
        df_right_threshold.to_csv(f'{self.results_path}/{self.dataset}/{self.config_file_name}/thresholds.csv')

        self.adata_right.obs['region'] = 'bad'
        self.adata_right.obs.loc[self.adata_right.obs['pathological_score'] < right_threshold, 'region'] = 'good'
        self.adata_right.obs['region'] = self.adata_right.obs['region'].astype('category')

        plt.close('all')
        plt.figure(figsize = (10, 10))
        plt.axis('off')
        plt.scatter(
            self.adata_right.obsm['spatial'][:, 0],
            self.adata_right.obsm['spatial'][:, 1],
            c=list(map(lambda x: 1 if x=='good' else 0,
                       pd.Categorical(self.adata_right.obs['region']))),
            cmap='plasma'
        )
        plt.savefig(f'{self.results_path}/{self.dataset}/{self.config_file_name}/segmentation_based_on_discrete_distribution.jpg')
        plt.close()

        pi_right_to_left = self.pi.T
        region_col = self.adata_right.obs['region'].values
        idx_adata_right_bad = np.where(region_col == 'bad')[0]
        
        col_sum = pi_right_to_left[idx_adata_right_bad].sum(axis=0)
        mapped_bad_idx_left_int = np.where(col_sum != 0)[0]
        idx_barcodes = self.adata_left.obs.index[mapped_bad_idx_left_int]
        self.adata_left.obs['region_mapped'] = 'good'
        self.adata_left.obs.loc[idx_barcodes, 'region_mapped'] = 'bad'

        sns.histplot(self.adata_right.obs['pathological_score'].values, kde=True, color="blue", ax=self.ax_hist_rs, bins=100)
        self.ax_hist_rs.legend(['Left (H)', 'Right (D)'])

        self.fig_hist_rs.savefig(f'{self.results_path}/{self.dataset}/{self.config_file_name}/rs_distribution_both_both_samples.jpg')

        if self.grid_search:
            if 'is_remodeled_for_grid_search' not in self.adata_right.obs.columns:
                print("Warning: 'is_remodeled_for_grid_search' column not found. Generating based on perturb logic.")
                # Import and use logic from perturb.py
                coor = self.adata_right.obsm['spatial']
                center_idx = random.randint(0, self.adata_right.n_obs - 1)
                rad = min(coor[:, 1].max() - coor[:, 1].min(), coor[:, 0].max() - coor[:, 0].min()) / 4
                remodeled_idxs = [idx for idx in range(self.adata_right.n_obs) if math.dist(coor[idx], coor[center_idx]) < rad]
                self.adata_right.obs['is_remodeled_for_grid_search'] = False
                self.adata_right.obs.loc[self.adata_right.obs.index[remodeled_idxs], 'is_remodeled_for_grid_search'] = True
            actual = self.adata_right.obs['is_remodeled_for_grid_search'].values
            predicted = np.array(list(map(lambda x: 1 if x == 'bad' else 0, self.adata_right.obs['region'].values)))

            F1_score = metrics.f1_score(actual, predicted)

            df_F1_score = pd.DataFrame({'F1_score': [F1_score]})
            df_F1_score.to_csv(f'{self.results_path}/{self.dataset}/{self.config_file_name}/F1_score.csv')


    def get_goodness_threshold_from_null_distribution(self, adata_left, adata_right=None):
        print("\nSynthesizing the healthy sample\n")
        if adata_right is None:
            adata_0, adata_1 = self.get_random_adatas_by_nearest_neighbor(adata_left)
        else:
            adata_0 = adata_left
            adata_1 = adata_right

        if self.use_gpu and torch.cuda.is_available():
            backend = ot.backend.TorchBackend()
            use_gpu = True
        else:
            backend = ot.backend.NumpyBackend()
            use_gpu = False

        if adata_right is None:
            cost_mat_path = f'{self.results_path}/../local_data/{self.dataset}/{self.sample_left}/cost_mat_{self.sample_left}_0_{self.sample_left}_1_{self.dissimilarity}.npy'
        else:
            cost_mat_path = f'{self.results_path}/../local_data/{self.dataset}/{self.sample_left}/cost_mat_Sham_1_Sham_2_{self.dissimilarity}.npy'

        if adata_right is None:
            morphology_cost_path = f'{self.results_path}/../local_data/{self.dataset}/{self.sample_left}/morphology_cost_{self.sample_left}_0_{self.sample_left}_1_{self.dissimilarity}.npy'
        else:
            morphology_cost_path = f'{self.results_path}/../local_data/{self.dataset}/{self.sample_left}/morphology_cost_Sham_1_Sham_2_{self.dissimilarity}.npy'
        
        os.makedirs(os.path.dirname(cost_mat_path), exist_ok=True)
        os.makedirs(os.path.dirname(morphology_cost_path), exist_ok=True)
        plt.switch_backend('agg')

        if not self.sinkhorn:
            print('sinkhorn not used')
        
        pi = paste_pairwise_align_modified(
            adata_0, adata_1,
            alpha=self.alpha, G_init=None, numItermax=10000,
            beta_morphology=self.beta_morphology,
            alpha_cell_spatial=self.alpha_cell_spatial,
            dissimilarity=self.dissimilarity, sinkhorn=self.sinkhorn,
            cost_mat_path=cost_mat_path, verbose=False, norm=True,
            morphology_cost_path=morphology_cost_path,
            backend=backend, use_gpu=use_gpu,
            numItermaxEmd=self.numIterMaxEmd
        )

        # Load both cost matrices and combine (same as alignment)
        cost_mat_gene = np.load(cost_mat_path)
        cost_mat_morph = np.load(morphology_cost_path)
        cost_mat = self._combine_cost_matrices(cost_mat_gene, cost_mat_morph, self.beta_morphology)


        distances_left, weights_left = compute_null_distribution(pi, cost_mat, 'left')
        print('\ndistances_left', distances_left.min(), distances_left.max())

        plt.figure(figsize=(9, 9))
        plt.tick_params(axis='x', labelsize=15)
        plt.tick_params(axis='y', labelsize=15)
        left_freqs = plt.hist(distances_left, bins=100)[0]
        
        os.makedirs(f'{self.results_path}/{self.dataset}/{self.config_file_name}/Histograms/', exist_ok=True)
        plt.savefig(f'{self.results_path}/{self.dataset}/{self.config_file_name}/Histograms/splitted_slice_left_pathological_score.jpg',format='jpg',dpi=350,bbox_inches='tight',pad_inches=0)

        distances_right, weights_right = compute_null_distribution(pi, cost_mat, 'right')
        print('distances_right', distances_right.min(), distances_right.max(), '\n')

        plt.figure(figsize=(9, 9))
        plt.tick_params(axis='x', labelsize=15)
        plt.tick_params(axis='y', labelsize=15)
        right_freqs = plt.hist(distances_right, bins=100)[0]

        os.makedirs(f'{self.results_path}/{self.dataset}/{self.config_file_name}/Histograms/', exist_ok=True)
        plt.savefig(f'{self.results_path}/{self.dataset}/{self.config_file_name}/Histograms/splitted_slice_right_pathological_score.jpg',format='jpg',dpi=350,bbox_inches='tight',pad_inches=0)

        ks_stat, p_value = stats.kstest(left_freqs, right_freqs)
        print('KS test statistic:', ks_stat, 'pvalue:', p_value)

        # Additional metrics
        ad_stat, ad_critical, ad_significance = stats.anderson_ksamp([left_freqs, right_freqs])
        print('Anderson-Darling test statistic:', ad_stat, 'significance level:', ad_significance)

        mw_stat, mw_p_value = stats.mannwhitneyu(left_freqs, right_freqs, alternative='two-sided')
        print('Mann-Whitney U test statistic:', mw_stat, 'pvalue:', mw_p_value)

        # Cohen's d effect size
        mean_left = np.mean(left_freqs)
        mean_right = np.mean(right_freqs)
        std_left = np.std(left_freqs, ddof=1)
        std_right = np.std(right_freqs, ddof=1)
        pooled_std = np.sqrt((std_left**2 + std_right**2) / 2)
        cohens_d = (mean_left - mean_right) / pooled_std if pooled_std != 0 else 0
        print("Cohen's d effect size:", cohens_d)

        # Combined metrics
        metrics_df = pd.DataFrame({
            'metric': ['KS_statistic', 'KS_p_value', 'AD_statistic', 'AD_significance_level', 'MW_U_statistic', 'MW_U_p_value', 'Cohens_d'],
            'value': [ks_stat, p_value, ad_stat, ad_significance, mw_stat, mw_p_value, cohens_d]
        })
        metrics_df.to_csv(f'{self.results_path}/{self.dataset}/{self.config_file_name}/metrics.csv', index=False)

        distances_both = np.array(list(distances_left) + list(distances_right))
        weights_both = np.array(list(weights_left) + list(weights_right))

        significance_threshold = 0.95
    
        bin_values_both = plt.hist(distances_both, weights=weights_both, bins = 100)
        pd.DataFrame({'Synthetic_spot_dist_both': distances_both}).to_csv(f'{self.results_path}/{self.dataset}/{self.config_file_name}/synthetic_spot_distances_both.csv')
        freqs = bin_values_both[0]
        print(f'max(bin_values_both[1]): {max(bin_values_both[1])}')
        sum_both = 0
        sum_tot = sum(freqs)
        for i in range(len(freqs)):
            sum_both += freqs[i]
            if sum_both/sum_tot > significance_threshold:
                both_threshold = bin_values_both[1][i]
                break
        sns.histplot(distances_both, kde=True, color="red", ax=self.ax_hist_rs, bins=100)
        
        return both_threshold


    def get_random_adatas_by_nearest_neighbor(self, adata):
        """
        Splits the AnnData object into two subsets by iteratively selecting random spots and their nearest unassigned neighbors.

        Args:
            adata: AnnData object with spatial coordinates in adata.obsm['spatial']

        Returns:
            adata_0, adata_1: Two AnnData subsets
        """
        coords = adata.obsm['spatial']
        n = len(coords)
        assigned = np.zeros(n, dtype=bool)
        group0 = []
        group1 = []

        from scipy.spatial import KDTree
        tree = KDTree(coords)

        while np.sum(~assigned) > 1:
            unassigned_indices = np.where(~assigned)[0]
            if len(unassigned_indices) == 0:
                break

            # Randomly pick one unassigned spot
            start = np.random.choice(unassigned_indices)
            assigned[start] = True

            # Find nearest unassigned neighbor
            dists, indices = tree.query(coords[start], k=n)
            nearest = None
            for neigh in indices:
                if not assigned[neigh] and neigh != start:
                    nearest = neigh
                    break

            if nearest is not None:
                assigned[nearest] = True
                group0.extend([start])
                group1.extend([nearest])
            else:
                # No unassigned neighbor, assign to smaller group
                if len(group0) <= len(group1):
                    group0.append(start)
                else:
                    group1.append(start)

        # Assign any remaining unassigned spots to the smaller group
        remaining = np.where(~assigned)[0]
        for r in remaining:
            if len(group0) <= len(group1):
                group0.append(r)
            else:
                group1.append(r)

        adata_0 = adata[group0]
        adata_1 = adata[group1]

        print(f'Group 0 size: {len(group0)}, Group 1 size: {len(group1)}')

        return adata_0, adata_1

    def get_2hop_adatas(self, adata):
        n = adata.obs['array_row'].max() + 1
        m = adata.obs['array_col'].max() + 1
        barcode_grid = np.empty([n, m], dtype='<U100')
        grid_idx = np.zeros((n, m)) - 1
        spot_rows = adata.obs['array_row']
        spot_cols = adata.obs['array_col']
        barcode_grid[spot_rows, spot_cols] = adata.obs.index
        grid_idx[spot_rows, spot_cols] = range(len(adata.obs.index))
        barcode_index = dict(zip(adata.obs.index, range(len(adata.obs.index))))

        col_max = grid_idx.max(axis=0)
        col_idxs = np.argwhere(col_max != -1).reshape(-1)

        grid_idx_0 = grid_idx[:, col_idxs[::2]]
        grid_idx_1 = grid_idx[:, col_idxs[1::2]]

        idxs_0_2hop = grid_idx_0[grid_idx_0 != -1].astype('int')
        idxs_1_2hop = grid_idx_1[grid_idx_1 != -1].astype('int')
        adata_0 = adata[idxs_0_2hop]
        adata_1 = adata[idxs_1_2hop]
        return adata_0, adata_1