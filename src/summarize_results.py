"""
Codes for summarizing results of the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd
from statistics import stdev


# configurations.
all_models = ['bpr', 'ubpr', 'wmf', 'expomf', 'relmf', 'upl_bpr', 'ubpr_nclip', 'relmf_du']
K = [3, 5, 8]
metrics = ['DCG', 'Recall', 'MAP']
col_names = [f'{m}@{k}' for m in metrics for k in K]
rel_col_names = [f'{m}@5' for m in metrics]

def str2float(str):
    return float(str)

def summarize_results(data: str, path: Path) -> None:
    """Load and save experimental results."""
    suffixes = ['all'] if data == 'coat' else ['cold-user', 'rare-item', 'all']
    for suffix in suffixes:
        aoa_list = []
        aoa_std_list = []
        for model in all_models:
            file = f'../logs/{data}/{model}/results/aoa_{suffix}.csv'
            aoa_list.append(pd.read_csv(file, index_col=0).mean(1))
            aoa_std_list.append(pd.read_csv(file, index_col=0).std(1))
        results_df = pd.concat(aoa_list, 1).round(7).T
        results_df.index = all_models
        results_df[col_names].to_csv(path / f'ranking_{suffix}.csv')
        results_std_df = pd.concat(aoa_std_list, 1).round(7).T
        results_std_df.index = all_models
        results_std_df[col_names].to_csv(path / f'ranking_{suffix}_std.csv')


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', '-d', required=True, nargs='*', type=str, choices=['coat', 'yahoo'])

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parser.parse_args()

    for data in args.datasets:
        path = Path(f'../paper_results/{data}')
        path.mkdir(parents=True, exist_ok=True)
        summarize_results(data=data, path=path)
