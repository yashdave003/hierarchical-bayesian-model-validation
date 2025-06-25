import os
import git
from pathlib import Path
from typing import List
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import plotly.graph_objects as go
from IPython.display import clear_output
import scipy
from scipy import stats
from scipy.spatial import ConvexHull
import pylustrator

ROOT_DIR = Path(git.Repo('.', search_parent_directories=True).working_tree_dir)

def variance_prior(r, eta, scale=1):
    beta = (eta+1.5)/r
    var_prior = scale * scipy.special.gamma(beta + 1/r)/scipy.special.gamma(beta)
    return var_prior

def kurtosis_prior(r, eta, fisher=True):
    beta = (eta+1.5)/r
    kurtosis = 3*scipy.special.gamma(beta + 2/r)*scipy.special.gamma(beta)/scipy.special.gamma(beta+1/r)**2 
    if fisher:
        return kurtosis - 3
    else:
        return kurtosis 

def find_master_dfs(root_dir: str) -> List[str]:
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    master_df_paths = []
    for current_dir, _, files in os.walk(root_path):
        if 'master_df.csv' in files:
            master_df_path = Path(os.path.join(current_dir, 'master_df.csv'))
            master_df_paths.append(str(master_df_path.absolute()))
    return master_df_paths

def add_hull(master_df, rEtaKsstats_dict, GROUP='group', debug=False):

    master_df_copy = master_df.copy()
    master_df_copy = master_df.set_index(GROUP)
    groups = master_df_copy.index
    master_df_copy["hull"] = ""

    for group in groups:
        if master_df_copy.loc[group, "total_samples"] < 10:
            master_df_copy.loc[group, "hull"] = np.nan
           
        else:
            drop_keys =list(rEtaKsstats_dict[group].keys())[-3:]
            if debug:
                print(drop_keys)
            pre_optimization = pd.DataFrame(rEtaKsstats_dict[group]).drop(drop_keys, axis = 1 )
            optimization = pd.DataFrame(rEtaKsstats_dict[group])[drop_keys]
            optimization = optimization.rename(columns = {"r_optimize": "r", "eta_optimize": "eta", drop_keys[-1]: "ksstat"})
            optimization = optimization.dropna()
            full_df = pre_optimization.merge(optimization, on=["r", "eta"], how="outer")
            full_df = full_df.set_index(["r", "eta"])
            full_df["ksstat"] = full_df.min(axis=1)
            full_df = full_df.reset_index()
            full_df = full_df[["r", "eta", "ksstat"]]
            full_df["1/beta"] = full_df["r"]/(full_df["eta"] + 1.5)
            MULT = 1.2
            cutoff = max(min(full_df["ksstat"]) * MULT, master_df_copy.loc[group, "kstest_stat_cutoff_0.05"], 0.01)
            filtered_df = full_df[full_df["ksstat"] < cutoff]
            points = np.column_stack((filtered_df["r"], filtered_df["1/beta"])) + stats.norm.rvs(size=(len(filtered_df), 2)) * 0.001  # Adding small noise for convex hull computation
            if len(points) < 3:
                hull=np.nan
            else:
                hull = ConvexHull(points)
            master_df_copy.loc[group, "hull"] = hull

    return master_df_copy.reset_index()



relevant_cols = [
        'group', 'obs_var', 'var_lower', 'var_upper', 'obs_kurt', 'kurt_lower', 'kurt_upper', 
        'total_samples', 'initial_r', 'initial_eta', 'kstest_stat_initial', 'kstest_stat_cutoff_0.05',
        'best_r', 'best_eta', 'kstest_stat_best',
        'param_gaussian', 'kstest_stat_gaussian', 'kstest_pval_gaussian', 
        'param_laplace', 'kstest_stat_laplace', 'kstest_pval_laplace', 
        'param_t', 'kstest_stat_t', 'kstest_pval_t', 'kstest_pval_gengamma', 
        'dataset', 'subset', 'transform', 'orientation', 'channel', 'github_plot', 'dataset_type', 'hull']

all_paths = find_master_dfs(os.path.join(ROOT_DIR, "results", "case-studies"))
all_master_dfs = []
github_plots_path = "https://github.com/yashdave003/hierarchical-bayesian-model-validation/blob/main/results/case-studies/"

for path in all_paths:
    if 'scaleTesting' in path or 'standardTesting' in path:
        continue
    master_df = pd.read_csv(path)
    master_df = master_df.rename(columns={master_df.columns[0]: 'group'})
    parts = Path(path).parts[-7:]
    if parts[0] == 'case-studies':
        parts = parts[1:]
    elif parts[0] == 'results':
        parts = parts[2:]
    if "MRI" in path:
        dataset, slice, transform, orientation, _, _ = parts
        master_df['dataset'] = dataset
        master_df['transform'] = transform
        master_df['subset'] = slice
        master_df['channel'] = np.nan
        master_df['orientation'] = orientation
        master_df['github_plot'] = [github_plots_path+f'{os.sep}'.join([dataset, slice, transform, orientation, 'plots', f'compare_cdf_pdf_layer_{group}.jpg']) for group in master_df['group']]
    elif len(parts) > 6:
        dataset, subset, transform, orientation, channel, _, _ = parts
        master_df['dataset'] = dataset
        master_df['transform'] = transform
        master_df['subset'] = subset
        master_df['channel'] = channel
        master_df['orientation'] = orientation
        master_df['github_plot'] = [github_plots_path+f'{os.sep}'.join([dataset, subset, transform, orientation, channel, 'plots', f'compare_cdf_pdf_layer_{group}.jpg']) for group in master_df['group']]
    elif "learned" in path:
        dataset, subset, transform, _, _ = parts
        master_df['dataset'] = dataset
        master_df['transform'] = transform
        master_df['subset'] = subset
        master_df = master_df.rename(columns={'filter_group' : 'orientation'})
        master_df['channel'] = np.nan
        master_df['github_plot'] = [github_plots_path+f'{os.sep}'.join([dataset, subset, transform, 'plots', f'compare_cdf_pdf_layer_{group}.jpg']) for group in master_df['group']]

    else:
        dataset, size, transform, channel, _, _ = parts
        master_df['dataset'] = dataset
        master_df['transform'] = transform
        master_df['subset'] = size
        master_df['channel'] = channel
        master_df['orientation'] = np.nan
        master_df['github_plot'] = [github_plots_path+f'{os.sep}'.join([dataset, size, transform, channel, 'plots', f'compare_cdf_pdf_layer_{group}.jpg']) for group in master_df['group']]
    
    if dataset in ['pastis', 'agriVision', 'spaceNet']:
        master_df['dataset_type'] = 'remote sensing'
    elif dataset in ['syntheticMRI2D', 'syntheticMRI3D']:
        master_df['dataset_type'] = 'medical'
    elif dataset in ['coco', 'segmentAnything']:
        master_df['dataset_type'] = 'natural'


    GROUP = 'layer' if transform.split("-")[0] == 'wavelet' else ('band' if transform.split("-")[0] == 'fourier' else 'filter_idx')
    rEtaKsstatsDict = pd.read_pickle(path[:-18] + "cache" + os.sep + "rEtaKsstats_dict.pickle")
    master_df = add_hull(master_df, rEtaKsstatsDict)
    all_master_dfs.append(master_df[relevant_cols])
    
main_df = pd.concat(all_master_dfs)

main_df['best_beta'] = (main_df['best_eta'] + 1.5)/main_df['best_r'] 
main_df['best_1/beta'] = 1/main_df['best_beta']
main_df['beat_all_priors'] = (main_df['kstest_stat_best'] < np.minimum.reduce([main_df['kstest_stat_gaussian'], main_df['kstest_stat_laplace'], main_df['kstest_stat_t']])).astype(int)
main_df["best_prior"] = np.array(["GenGamma", "Gaussian", "Laplace", "Student-T", np.nan])[
                                np.nanargmin(np.array([main_df['kstest_stat_best'], 
                                                        main_df['kstest_stat_gaussian'], 
                                                        main_df['kstest_stat_laplace'], 
                                                        main_df['kstest_stat_t'], 
                                                        0.99*np.ones_like(main_df['kstest_stat_t'])]
                                                        ).T, axis=1)]

# frequency_map = pd.read_csv(os.path.join(ROOT_DIR, "transformed-data", "master-frequency-map.csv")).set_index(['dataset', 'transform', 'group'])
# main_df = main_df.set_index(['dataset', 'subset', 'transform', 'group']).merge(frequency_map, left_index = True, right_index=True).reset_index()

# Download existing results_categorization_sheet and place in publication/paper/code
# Merge completed manually classified fits with existing table

old_fail_cat_df_path = os.path.join(ROOT_DIR, "publication", "paper", "CSVs", 'result_categorization_sheet - combined_categories.csv')
old_fail_cat_df = pd.read_csv(old_fail_cat_df_path)
main_df = main_df.merge(old_fail_cat_df[['github_plot', 'failure_category', 'failure_type', 'which_ones']], on='github_plot', how='left')

# Save df after adding new runs of testing pipeline
save_cols = ['total_samples', 'dataset', 'subset', 'transform', 'orientation', 'channel', 'group', 'kstest_stat_best', 'kstest_stat_cutoff_0.05', 'beat_all_priors', 'failure_category', 'failure_type', 'which_ones', 'github_plot']
new_fail_cat_df_path = os.path.join(ROOT_DIR, "publication", "paper", "CSVs", 'new_fail_cat_df.csv')
main_df[save_cols].to_csv(new_fail_cat_df_path)
main_df = main_df.reset_index(drop=True)

