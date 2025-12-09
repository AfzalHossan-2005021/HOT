import os
import ot
import torch
import numpy as np
import scanpy as sc
from DataLoader import DataLoader
from utils import scale_coords, QC, paste_pairwise_align_modified


class PairwiseAlign():
    def __init__(self, config):
        self.config = config
        self.dataset = config['dataset']
        self.sample_left = config['sample_left']
        self.sample_right = config['sample_right']
        self.alpha = config['alpha']
        self.dissimilarity = config['dissimilarity']
        self.init_map_scheme = config['init_map_scheme']
        self.numIterMaxEmd = config['numIterMaxEmd']
        self.numInnerIterMax = config['numInnerIterMax']
        self.use_gpu = config['use_gpu']
        self.results_path = config['results_path']
        self.config_file_name = os.path.basename(config['config_path'])
        self.sinkhorn = config['sinkhorn']
        self.lambda_sinkhorn = config['lambda_sinkhorn']
        self.beta_morphology = config['beta_morphology']
        self.alpha_cell_spatial = config.get('alpha_cell_spatial', 0.3)  # Default to 0.3 for backward compatibility
        self.cost_mat_path = f'{self.results_path}/../local_data/{self.dataset}/{self.sample_left}/cost_mat_{self.sample_left}_{self.sample_right}_{self.dissimilarity}.npy'
        self.morphology_cost_path = f'{self.results_path}/../local_data/{self.dataset}/{self.sample_left}/morphology_cost_{self.sample_left}_{self.sample_right}_{self.dissimilarity}.npy'
        os.makedirs(os.path.dirname(self.cost_mat_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.morphology_cost_path), exist_ok=True)

        if config['adata_left_path'] != 'None':
            self.adata_left = sc.read(config['adata_left_path'])
            self.adata_right = sc.read(config['adata_right_path'])
        else:
            data_loader = DataLoader(config)

            dataset_map = data_loader.read_data(self.dataset)

            self.adata_left = dataset_map[self.sample_left]
            self.adata_right = dataset_map[self.sample_right]

        scale_coords(self.adata_left, key_name='spatial')
        scale_coords(self.adata_right, key_name='spatial')

        if config['QC']:
            QC(self.adata_left)
            QC(self.adata_right)

    def pairwise_align_sinkhorn(self):
        if not torch.cuda.is_available():
            if self.use_gpu:
                print("Setting use_gpu to False")
            self.use_gpu = False

        if self.use_gpu:
            backend = ot.backend.TorchBackend()
        else:
            backend = ot.backend.NumpyBackend()
        pi_init = None

        if self.init_map_scheme == 'uniform':
            pi_init = None
            
        print("Calculating pi using gcg")

        self.config['cost_mat_path'] = self.cost_mat_path
        self.config['morphology_cost_path'] = self.morphology_cost_path

        if self.lambda_sinkhorn == 'inf':
            pi = np.ones((self.adata_left.n_obs, self.adata_right.n_obs)) / (self.adata_left.n_obs * self.adata_right.n_obs)
            if self.use_gpu:
                pi = torch.from_numpy(pi)
            return pi, -1000
        
        pi, fgw_dist = paste_pairwise_align_modified(
            self.adata_left, self.adata_right,
            alpha=self.alpha, sinkhorn=self.sinkhorn,
            lambda_sinkhorn=self.lambda_sinkhorn,
            beta_morphology=self.beta_morphology,
            alpha_cell_spatial=self.alpha_cell_spatial,
            dissimilarity=self.dissimilarity, G_init=pi_init,
            numItermax=10000, cost_mat_path=self.cost_mat_path,
            morphology_cost_path=self.morphology_cost_path,
            return_obj=True, norm=True, verbose=False,
            backend=backend, use_gpu=self.use_gpu,
            numInnerItermax=self.numInnerIterMax, method='sinkhorn_log'
        )
        return pi, fgw_dist