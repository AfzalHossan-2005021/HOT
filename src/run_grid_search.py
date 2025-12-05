import os
import json
import argparse

parser = argparse.ArgumentParser(prog='SPaSE')
parser.add_argument('-ifp', '--data_folder_path', default='../Data')

parser.add_argument('-d', '--dataset')
parser.add_argument('-l', '--left_sample_name')
parser.add_argument('-r', '--right_sample_name')

parser.add_argument('-hr', '--healthy_right_sample_name', default='None')
parser.add_argument('-g', '--use_gpu', default=1)

parser.add_argument('-ds', '--dissimilarity', default='js')
parser.add_argument('-sh', '--sinkhorn', default=1)
parser.add_argument('-ims', '--init_map_scheme', default='uniform')
parser.add_argument('-nim', '--numInnerIterMax', default=10000)
parser.add_argument('-nem', '--numIterMaxEmd', default=1000000)
parser.add_argument('-rp', '--results_path', default='../results')
parser.add_argument('-qc', '--QC', default=0)


args = parser.parse_args()

alphas = [0.0001, 0.001, 0.01, 0.1]
lambda_sinkhorns = [0.001, 0.01, 0.1]
beta_morphology = [0.0, 0.25, 0.5, 0.75, 1.0]

for alpha in alphas:
    for lambda_sinkhorn in lambda_sinkhorns:
        for beta in beta_morphology:
            config = {
                "grid_search": 1,
                "alpha": alpha,
                "lambda_sinkhorn": lambda_sinkhorn,
                "beta_morphology": beta,
                "mode": 1,
                "data_folder_path": args.data_folder_path,
                "sample_left": args.left_sample_name,
                "dataset": args.dataset,
                "sample_right": args.right_sample_name,
                "adata_left_path": f'{args.data_folder_path}/{args.dataset}/{args.left_sample_name}.h5ad',
                "adata_right_path": f'{args.data_folder_path}/{args.dataset}/{args.right_sample_name}.h5ad',
                "adata_healthy_right_path": f'{args.data_folder_path}/{args.dataset}/{args.healthy_right_sample_name}.h5ad' if args.healthy_right_sample_name != 'None' else 'None',
                "sinkhorn": int(args.sinkhorn),
                "dissimilarity": args.dissimilarity,
                "init_map_scheme": args.init_map_scheme,
                "numIterMaxEmd": int(args.numIterMaxEmd),
                "numInnerIterMax": int(args.numInnerIterMax),
                "use_gpu": int(args.use_gpu),
                "QC": int(args.QC),
                "results_path": args.results_path,
            }

            config_file_name = f'config_{args.dataset}_{args.left_sample_name}_vs_{args.right_sample_name}_{args.dissimilarity}'
            if int(args.sinkhorn):
                config_file_name += f'_sinkhorn_lambda_{lambda_sinkhorn}_alpha_{alpha}_beta_{beta}.json'
            else:
                config_file_name += f'_alpha_{alpha}_beta_{beta}.json'

            config_path = f'../configs/{config_file_name}'

            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            with open(config_path, 'w') as config_file:
                json.dump(config, config_file, indent=4)

            os.system(f'python main.py --config {config_path}')

            with open(config_path) as f:
                config = json.load(f)

            config['mode'] = 2

            with open(config_path, 'w') as config_file:
                json.dump(config, config_file, indent=4)

            os.system(f'python main.py --config {config_path}')