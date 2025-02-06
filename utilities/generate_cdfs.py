from logging import raiseExceptions
from signal import raise_signal
import os
import git
from pathlib import Path
import os
ROOT_DIR = Path(git.Repo('.', search_parent_directories=True).working_tree_dir)

os.chdir(os.path.join(ROOT_DIR, "utilities"))
from testing import *
os.chdir(os.path.join(ROOT_DIR, "results"))

def simple_add_cdfs(r_range, eta_range, folder_name = '', n_samples = 2000, tail_bound = 0.01, tail_percent = 0.1, use_matlab=False, eng=None, enforce_assert=True, return_assert = False, debug=False):

    if not os.path.isdir("CDFs"):
        raise Exception("This Directory Does Not Contain CDFs")
    
    if folder_name == '':
        folder_name = f'r{round_to_sigfigs(min(r_range))}to{round_to_sigfigs(max(r_range))}_eta{round_to_sigfigs(min(eta_range))}to{round_to_sigfigs(max(eta_range))}'

    FOLDER_PATH = os.path.join("CDFs", folder_name)

    if os.path.isdir(FOLDER_PATH):
        cdfs_completed = combine_pickles(os.path.join("CDFs", folder_name))

        print("CDFs completed:", len(cdfs_completed))
    else:
        Path(os.path.join(os.getcwd(), FOLDER_PATH)).mkdir()
        cdfs_completed = dict()

    n = len(r_range)*len(eta_range)

    cnt = len(cdfs_completed)
    for r in r_range:
        r_cdf = dict()
        r = round_to_sigfigs(r)
        for eta in eta_range:
            eta = round_to_sigfigs(eta)
            if ((r, eta) in cdfs_completed) and cdfs_completed[(r, eta)]:
                continue
            cnt += 1

            print(f'{(r, eta)}, {cnt} of {n}')
  
            computed_cdf = compute_prior_cdf(r = r, eta = eta, method = 'gamma_cdf', n_samples = n_samples, tail_percent = tail_percent, tail_bound = tail_bound, 
                                             use_matlab=use_matlab, eng=eng, enforce_assert=enforce_assert, return_assert=return_assert, debug=debug)
            if computed_cdf is None:
                with open("generate_CDF_log.csv", 'a') as handle:
                    handle.write(f"{r}, {eta}, {n_samples}, failed assert\n")
                continue
            r_cdf[(r, eta)] = computed_cdf
        if r_cdf:
            sorted_r_cdf = [i[1] for i in sorted(r_cdf)]
            min_eta, max_eta = round_to_sigfigs(min(sorted_r_cdf), 6), round_to_sigfigs(max(sorted_r_cdf), 6)
            pkl_path = os.path.join(FOLDER_PATH, f'r{r}_eta{min_eta}-{max_eta}.pickle')
            pd.to_pickle(r_cdf, pkl_path)
        else:
            print(f"Skipped {r} entirely")

    if debug:
        print(f'You can find the CDFs here: {os.path.join(os.getcwd(), FOLDER_PATH)}')

all_eta = np.arange(-1.45, -1, 0.01)
all_r = np.append(np.arange(0.1, 2, 0.1), np.arange(2, 10, 1))
n_samples = 1000
tail_percent = 0.1
tail_bound = 1e-5

simple_add_cdfs(all_r, all_eta, n_samples = n_samples, folder_name='',
                tail_percent = tail_percent, tail_bound = tail_bound, use_matlab=True, 
                eng=eng, enforce_assert=False, return_assert=True, debug=True)

all_eta = np.arange(-1, -0.5, 0.01)
all_r = np.append(np.arange(0.1, 2, 0.1), np.arange(2, 10, 1))
n_samples = 1000
tail_percent = 0.1
tail_bound = 1e-5

simple_add_cdfs(all_r, all_eta, n_samples = n_samples, folder_name='',
                tail_percent = tail_percent, tail_bound = tail_bound, use_matlab=True, 
                eng=eng, enforce_assert=False, return_assert=True, debug=True)

all_eta = np.arange(-0.5, 0, 0.01)
all_r = np.append(np.arange(0.1, 2, 0.1), np.arange(2, 10, 1))
n_samples = 1000
tail_percent = 0.1
tail_bound = 1e-5

simple_add_cdfs(all_r, all_eta, n_samples = n_samples, folder_name='',
                tail_percent = tail_percent, tail_bound = tail_bound, use_matlab=True, 
                eng=eng, enforce_assert=False, return_assert=True, debug=True)

if USE_MATLAB:
    eng.quit()
