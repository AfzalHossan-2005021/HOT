import os
import json
import argparse
import numpy as np
import pandas as pd
from PairwiseAlign import PairwiseAlign
from AnalyzeOutput import AnalyzeOutput


def main():
    config = load_config()

    print("\nConfiguration:")
    print("sample_left: ", config['sample_left'])
    print("sample_right: ", config['sample_right'])
    print('alpha: ', config['alpha'])
    print('lambda: ', config['lambda_sinkhorn'])
    print('-' * 40)

    mode = config['mode']

    if mode == 1:
        print('mode == 1, doing pairwise align')
        pairwise_align_model = PairwiseAlign(config)

        pi, fgw_dist = pairwise_align_model.pairwise_align_sinkhorn()
        config['fgw_dist'] = float(fgw_dist)
        dissimilarity = config['dissimilarity']
        
        pi_file_name = f"{config['dataset']}_{config['init_map_scheme']}_{dissimilarity}.npy"

        config_file_name = os.path.basename(config['config_path'])
        config['pi_path'] = f"{config['results_path']}/{config['dataset']}/{config_file_name}/Pis/{pi_file_name}"
        
        os.makedirs(f"{config['results_path']}/{config['dataset']}/{config_file_name}/Pis/", exist_ok=True)
        np.save(config['pi_path'], pi)
        
        with open(config['config_path'], 'w') as config_file:
            json.dump(config, config_file, indent=4)

    if mode == 2:
        print('\nmode == 2, analyzing output')
        if 'pi_path' not in config:
            print("mode == 2, but pi path not set")
            return
        pi = np.load(config['pi_path'])
        config['pi'] = pi
        output_analyzer = AnalyzeOutput(config)
        
        output_analyzer.visualize_goodness_of_mapping(slice_pos='right')
        
        output_analyzer.divide_into_2_regions_wrt_goodness_score_and_find_DEG()

        pd.DataFrame({'fgw_distance': [config['fgw_dist']]}).to_csv(f"{config['results_path']}/{config['dataset']}/{os.path.basename(config['config_path'])}/fgw_dist.csv")

def load_config():
    parser = argparse.ArgumentParser(description='Modified PASTE pipeline')
    parser.add_argument('--config', help="The input parameters json file path", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    config['config_path'] = args.config
    return config


if __name__ == "__main__":
    main()