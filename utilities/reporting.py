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
from scipy.spatial import Delaunay
from scipy.spatial import distance
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from tqdm import tqdm

ROOT_DIR = Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
RERUN=False
DPI=300

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

def line_intersects_hull(hull):

    x_vals = np.linspace(0, 20, 1000)
    eta_vals = 1.5 + np.zeros_like(x_vals)
    roi_beta = (eta_vals) / x_vals
    roi = 1 / roi_beta
    
    if hull is None or hull is np.nan:
        return 0
    line_points = np.column_stack((x_vals, roi))
    return int(np.any(in_hull(line_points, hull)))

def add_hull_and_kurt(master_df, rEtaKsstats_dict, GROUP='group', debug=False):

    master_df_copy = master_df.copy()
    master_df_copy = master_df.set_index(GROUP)
    groups = master_df_copy.index
    master_df_copy["hull"] = ""

    for group in groups:
        obs_kurt, kurt_lower, kurt_upper = master_df_copy.loc[group, "obs_kurt"], master_df_copy.loc[group, "kurt_lower"], master_df_copy.loc[group, "kurt_upper"]
        best_scale = master_df_copy.loc[group, "best_scale"]

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
            full_df['kurt'] = full_df.apply(lambda row : kurtosis_prior(r=row['r'], eta=row['eta'], scale=best_scale), axis=1)
            full_df['pass_kurt'] = (full_df['kurt'] > kurt_lower) & (full_df['kurt'] < kurt_upper)
            kurt_df = full_df.copy()
            full_df = full_df.set_index(["r", "eta"])
            full_df["ksstat"] = full_df.min(axis=1)
            full_df = full_df.reset_index()
            full_df = full_df[["r", "eta", "ksstat"]]
            full_df["1/beta"] = full_df["r"]/(full_df["eta"] + 1.5)
            MULT = 1.2
            cutoff = max(min(full_df["ksstat"]) * MULT, 0.01)
            filtered_df = full_df[full_df["ksstat"] < cutoff]
            points = np.column_stack((filtered_df["r"], filtered_df["1/beta"])) + stats.norm.rvs(size=(len(filtered_df), 2)) * 0.001  # Adding small noise for convex hull computation
            if len(points) < 3:
                hull=np.nan
            else:
                hull = ConvexHull(points)
            master_df_copy.loc[group, "hull"] = hull

            master_df_copy.loc[group, "pass_kurt_anywhere"] = np.any(kurt_df['pass_kurt'] > 0)
            master_df_copy.loc[group, "pass_kurt_intersect_hull"] = np.any((kurt_df['pass_kurt'] == 1) & (kurt_df['ksstat'] < cutoff))

    return master_df_copy.reset_index()

def in_hull(p, hull):
    if hasattr(hull, 'vertices') and not isinstance(hull, Delaunay):
        hull = Delaunay(hull.points)
    elif not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

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
            cutoff = max(min(full_df["ksstat"]) * MULT, 0.01)
            filtered_df = full_df[full_df["ksstat"] < cutoff]
            points = np.column_stack((filtered_df["r"], filtered_df["1/beta"])) + stats.norm.rvs(size=(len(filtered_df), 2)) * 0.001  # Adding small noise for convex hull computation
            if len(points) < 3:
                hull=np.nan
            else:
                hull = ConvexHull(points)
            master_df_copy.loc[group, "hull"] = hull

    return master_df_copy.reset_index()

save_path_with_hull = Path(os.path.join(ROOT_DIR, 'publication', 'paper', 'CSVs', 'final_results_with_hull.pickle'))
save_path_without_hull = Path(os.path.join(ROOT_DIR, 'publication', 'paper', 'CSVs', 'final_results.csv'))

if RERUN or save_path_without_hull.exists():
    main_df = pd.read_csv(save_path_without_hull).drop("Unnamed: 0", axis=1)
    main_df = main_df[main_df['total_samples'] > 100]
    main_df['best_beta'] = (main_df['best_eta'] + 1.5)/main_df['best_r']
    main_df = main_df#[(main_df['dataset'] != 'standardTesting')]
else:
    print("Run make_results_csv.ipynb first")






###HULL REGENERATION
def process_rEtaKsstats_dict(master_df_original, rEtaKsstats_dict, sample_limit = 10, GROUP='group', debug=False):

    master_df = master_df_original.copy()
    master_df = master_df_original.set_index(GROUP)
    groups = master_df.index
    master_df["hull"] = ""

    for group in groups:
        if master_df.loc[group, "total_samples"] < sample_limit:
            master_df.loc[group, "hull"] = np.nan
           
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
            significant_ksstat = master_df.loc[group, "kstest_stat_cutoff_0.05"]
            #cutoff = max(min(full_df["ksstat"]) * MULT, 0.01)
            cutoff = max(significant_ksstat, 0.01)
            if min(full_df["ksstat"]) * MULT > 0.01:
                uses_practical_threshold = 0
            else:
                uses_practical_threshold = 1

            master_df.loc[group, "use_practical_threshold"] = uses_practical_threshold 
            
            filtered_df = full_df[full_df["ksstat"] < cutoff]
            points = np.column_stack((filtered_df["r"], filtered_df["1/beta"])) + stats.norm.rvs(size=(len(filtered_df), 2)) * 0.001  # Adding small noise for convex hull computation

            if len(points) < 3:
                hull=np.nan
                hull_area = 0
                intersect_roi = 0
            else:
                hull = ConvexHull(points)
                hull_area = hull.volume

                if np.any(filtered_df["eta"] > 0) and np.any(filtered_df["eta"] < 0):
                    intersect_roi = 1
                else:
                    intersect_roi = 0

            master_df.loc[group, "hull"] = hull    
            master_df.loc[group, "hull_area"] = hull_area
            master_df.loc[group, "intersect_roi"] = intersect_roi

            master_df.loc[group, "hull_r_lower"] = filtered_df['r'].min()
            master_df.loc[group, "hull_r_upper"] = filtered_df['r'].max()
            master_df.loc[group, "hull_beta_lower"] = 1/filtered_df['1/beta'].max()
            master_df.loc[group, "hull_beta_upper"] = 1/filtered_df['1/beta'].min()

            # print(master_df.loc[group, "hull_r_lower"], master_df.loc[group, "hull_r_upper"], master_df.loc[group, "hull_beta_lower"], master_df.loc[group, "hull_beta_upper"])

            # kurt_lower, kurt_upper = master_df.loc[group, "kurt_lower"], master_df.loc[group, "kurt_upper"]
            # best_scale = master_df.loc[group, "best_scale"]

            # kurt_df = full_df.copy()
            # kurt_df['kurt'] = kurt_df.apply(lambda row : kurtosis_prior(r=row['r'], eta=row['eta'], scale=best_scale), axis=1)
            # kurt_df['pass_kurt'] = (kurt_df['kurt'] > kurt_lower) & (kurt_df['kurt'] < kurt_upper)

            # pass_kurt_anywhere = np.sum(kurt_df['pass_kurt'] > 0)
            # if pass_kurt_anywhere >= 3:
            #     temp = kurt_df[kurt_df['pass_kurt'] == 1]
            #     points_kurt = np.column_stack((temp["r"], temp["1/beta"])) + stats.norm.rvs(size=(len(temp), 2)) * 0.001
            #     hull_kurt = ConvexHull(points_kurt)
            #     hull_kurt_area = hull_kurt.volume
            # else:
            #     hull_kurt = np.nan
            #     hull_kurt_area = 0

            # master_df.loc[group, "hull_kurt"] = hull_kurt
            # master_df.loc[group, "hull_kurt_area"] = hull_kurt_area
            # master_df.loc[group, "num_pass_kurt_anywhere"] = pass_kurt_anywhere
            # master_df.loc[group, "pass_kurt_anywhere"] = int(pass_kurt_anywhere > 0)
            # master_df.loc[group, "num_pass_kurt_intersect_hull"] = np.sum((kurt_df['pass_kurt'] == 1) & (kurt_df['ksstat'] < cutoff))
            # master_df.loc[group, "pass_kurt_intersect_hull"] = int(master_df.loc[group, "num_pass_kurt_intersect_hull"] > 0)

    return master_df.reset_index()

def regenerate_hulls(sample_limit =10):
   

    relevant_cols = [
            'group', 'dataset', 'subset', 'transform', 'orientation', 'channel', 'dataset_type', 
            'obs_var', 'var_lower', 'var_upper', 
            'total_samples', 'initial_r', 'initial_eta',  'best_r', 'best_eta', 'best_scale',
            'kstest_stat_initial', 'kstest_stat_cutoff_0.05', 'kstest_stat_best', 'n_pval_0.05', 
            'obs_kurt', 'kurt_lower', 'kurt_upper', 
            'intersect_roi', 'hull', 'hull_area',
            'hull_r_lower', 'hull_r_upper', 'hull_beta_lower', 'hull_beta_upper', 
            # 'num_pass_kurt_anywhere', 'pass_kurt_anywhere', 'num_pass_kurt_intersect_hull', 'pass_kurt_intersect_hull', 'hull_kurt', 'hull_kurt_area',
            'param_gaussian', 'kstest_stat_gaussian', 'kstest_pval_gaussian', 
            'param_laplace', 'kstest_stat_laplace', 'kstest_pval_laplace', 
            'param_t', 'kstest_stat_t', 'kstest_pval_t', 'kstest_pval_gengamma', 
            'github_plot']

    all_paths = find_master_dfs(os.path.join(ROOT_DIR, "results", "case-studies"))
    all_master_dfs = []
    github_plots_path = "https://github.com/yashdave003/hierarchical-bayesian-model-validation/blob/main/results/case-studies/"

    for i in tqdm(range(len(all_paths))):
        path = all_paths[i]
        if 'scaleTesting' in path:
            continue
        if 'gabor' in path:
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
            master_df['github_plot'] = [github_plots_path+'/'.join([dataset, slice, transform, orientation, 'plots', f'compare_cdf_pdf_layer_{group}.jpg']) for group in master_df['group']]
        elif len(parts) > 6:
            dataset, subset, transform, orientation, channel, _, _ = parts
            master_df['dataset'] = dataset
            master_df['transform'] = transform
            master_df['subset'] = subset
            master_df['channel'] = channel
            master_df['orientation'] = orientation
            master_df['github_plot'] = [github_plots_path+'/'.join([dataset, subset, transform, orientation, channel, 'plots', f'compare_cdf_pdf_layer_{group}.jpg']) for group in master_df['group']]
        elif "learned" in path:
            dataset, subset, transform, _, _ = parts
            master_df['dataset'] = dataset
            master_df['transform'] = transform
            master_df['subset'] = subset
            master_df = master_df.rename(columns={'filter_group' : 'orientation'})
            master_df['channel'] = np.nan
            master_df['github_plot'] = [github_plots_path+'/'.join([dataset, subset, transform, 'plots', f'compare_cdf_pdf_layer_{group}.jpg']) for group in master_df['group']]

        else:
            dataset, size, transform, channel, _, _ = parts
            master_df['dataset'] = dataset
            master_df['transform'] = transform
            master_df['subset'] = size
            master_df['channel'] = channel
            master_df['orientation'] = np.nan
            master_df['github_plot'] = [github_plots_path+'/'.join([dataset, size, transform, channel, 'plots', f'compare_cdf_pdf_layer_{group}.jpg']) for group in master_df['group']]
        
        if dataset in ['pastis', 'agriVision', 'spaceNet']:
            master_df['dataset_type'] = 'remote sensing'
        elif dataset in ['syntheticMRI2D', 'syntheticMRI3D']:
            master_df['dataset_type'] = 'medical'
        elif dataset in ['coco', 'segmentAnything', 'standardTesting']:
            master_df['dataset_type'] = 'natural'
        elif dataset in ['standardTesting']:
            master_df['dataset_type'] = 'classical'

        GROUP = 'layer' if transform.split("-")[0] == 'wavelet' else ('band' if transform.split("-")[0] == 'fourier' else 'filter_idx')
        rEtaKsstatsDict = pd.read_pickle(path[:-18] + "cache" + os.sep + "rEtaKsstats_dict.pickle")
        
        master_df = process_rEtaKsstats_dict(master_df, rEtaKsstatsDict, sample_limit = sample_limit)
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

    print("Main DF (before frequency):", main_df.shape)
    frequency_map = pd.read_csv(os.path.join(ROOT_DIR, "transformed-data", "master-frequency-map.csv")).set_index(['dataset', 'transform', 'group'])
    main_df = main_df.set_index(['dataset', 'subset', 'transform', 'group']).merge(frequency_map, left_index = True, right_index=True, how='left').reset_index()

    print("Main DF (after frequency):", main_df.shape)
    old_fail_cat_df_path = os.path.join(ROOT_DIR, "publication", "paper", "CSVs", 'result_categorization_sheet - combined_categories.csv')
    old_fail_cat_df = pd.read_csv(old_fail_cat_df_path)
    main_df = main_df.merge(old_fail_cat_df[['github_plot', 'failure_category', 'failure_type', 'which_ones']], on='github_plot', how='left')
    print("Main DF (after result categorization):", main_df.shape)
    return main_df