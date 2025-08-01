import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from testing import *

GROUP_NAME = 'Group (Layer/Band)'

def get_random_color():
    return np.random.choice(list(mcolors.CSS4_COLORS.values()))

def combo_test_plot(df, cols, extra_boundary = 0.5, plot_name = '', target_var = None, best_param=None, best_ksstat=None):
    cols = sorted(cols)
    df = df.copy() 

    for col in cols:
        df[col] = df[col].replace({True: col[5:], False: ''})
    df['map'] = df.apply(lambda row : ''.join([row.loc[col].capitalize() +'' for col in cols]), axis = 1)
    temp = df[(df['map'] != '') & (df['map'] != 'Var')&(df['map'] != 'Kurt') & (df['map'] != 'KurtVar')]
    
    if len(temp) == 0:
        df = df[(df['r'] >= best_param[0] - extra_boundary) & 
            (df['r'] <= best_param[0] + extra_boundary) &
            (df['eta'] >= best_param[1] - extra_boundary) & 
            (df['eta'] <= best_param[1] + extra_boundary)]
    else:
        df = df[(df['r'] >= temp['r'].min() - extra_boundary) & 
            (df['r'] <= temp['r'].max() + extra_boundary) &
            (df['eta'] >= temp['eta'].min() - extra_boundary) & 
            (df['eta'] <= temp['eta'].max() + extra_boundary)]
    fixed_palette = {
    'Var': 'xkcd:dark yellow',
    'Kstest': 'blue',
    'Kurt': 'orange',
    'KstestVar': 'cyan',
    'KurtVar': 'red',
    'KstestKurt': 'brown',
    'KstestKurtVar': 'xkcd:shamrock green',
    '': 'xkcd:medium gray'
}
    map_categories = df['map'].unique()
    for m in map_categories:
        if m not in fixed_palette:
            fixed_palette[m] = get_random_color()
    fig, ax = plt.subplots()
    fig = sns.scatterplot(df, x='r', y='eta', hue='map', palette = fixed_palette, ax=ax, alpha=1, edgecolor='none')
    r_vals = []
    eta_vals = []
    if target_var:
        for r in np.linspace(0.1, df['r'].max() if df.shape[0] > 0 else 100, 1000):
            eta = find_eta_for_target_mean(r, target_var)
            if eta < df['eta'].min():
                continue
            if (eta > df['eta'].max()):
                break
            r_vals.append(r)
            eta_vals.append(eta)
        sns.lineplot(x=r_vals, y=eta_vals, label=f'target_var:{np.round(target_var, 4)}', ax=ax)
    if best_param is not None:
        sns.scatterplot(x = [best_param[0]], y = [best_param[1]], marker='*', s = 60, c = 'xkcd:electric pink', ax=ax, label = f'Best: {best_param}', edgecolor='none')
        if best_ksstat is not None:
            sns.scatterplot(x = [best_param[0]], y = [best_param[1]], marker='*', s = 1, c = 'xkcd:electric pink', ax=ax, label = f'KS Stat: {np.round(best_ksstat, decimals=5)}', edgecolor='none')
    plt.legend(loc = 'upper right')
    if plot_name:
        plt.title(plot_name)
    else:
        plt.title(f"{', '.join([col[5:].capitalize() for col in cols])} with boundary {extra_boundary}")
    if len(df) > 0:
        create_scatter_plot(df=df, metric='ksstat', plot_name=f"KS Stats + {plot_name if plot_name else ''}", log_colorbar=True) 
    
    plt.show()
    return fig

def create_scatter_plot(df, metric=None, plot_name = '', log_colorbar=False):
    """
    Create a scatter plot, where the color of each point represents the value from the specified metric column.
    If metric=None, plot all the (r, eta) values in df.

    Arguments:
    df : A pandas DataFrame containing the columns 'r', 'eta', and the specified metric column.
    metric : The name of the column in the DataFrame to use for color mapping.
    log_scale : Boolean, if True, the color scale of the plot will be logarithmic.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    if metric:
        if pd.api.types.is_numeric_dtype(df[metric]):
            norm = LogNorm() if log_colorbar else None
            scatter = ax.scatter(df['r'], df['eta'], c=df[metric], cmap='viridis', alpha=1, norm=norm)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(metric)
        else:
            categories = df[metric].unique()
            color_map = plt.colormaps['accent']
            colors = {cat: color_map(i/len(categories)) for i, cat in enumerate(categories)}
            
            for cat in categories:
                mask = df[metric] == cat
                ax.scatter(df.loc[mask, 'r'], df.loc[mask, 'eta'], 
                           c=[colors[cat]], label=cat, alpha=1)
            
            ax.legend(title=metric)
        if plot_name:
            ax.set_title(plot_name)
        else:
            ax.set_title(f'(r, eta) pairs colored by {metric}')
    else:
        sns.scatterplot(x=df['r'], y=df['eta'], color='xkcd:gray', alpha=0.5, ax=ax, edgecolor='none')
        ax.set_title('(r, eta) pairs for which CDFs are computed (Linear eta)')

    ax.set_xlabel('r')
    ax.set_ylabel('eta')
    plt.grid(which='both')
    plt.show()

    return fig


def create_scatter_plots(df, metric1, metric2):
    """
    Create two scatter plots side by side, where the color of each point in the first plot represents the value from the first specified metric column, and the color of each point in the second plot represents the value from the second specified metric column.

    Arguments:
    df : A pandas DataFrame containing the columns 'r', 'eta', and the specified metric columns.
    metric1 : The name of the column in the DataFrame to use for color mapping in the first plot.
    metric2 : The name of the column in the DataFrame to use for color mapping in the second plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1 (Linear eta, colored by metric1)
    scatter1 = ax1.scatter(df['r'], df['eta'], c=df[metric1], cmap='viridis', alpha=0.6, edgecolor='none')
    plt.grid(True)
    ax1.set_xlabel('r')
    ax1.set_ylabel('eta')
    ax1.set_title(f'(r, eta) pairs colored by {metric1}')
    cbar1 = fig.colorbar(scatter1, ax=ax1)
    cbar1.set_label(metric1)
    

    # Plot 2 (Linear eta, colored by metric2)
    scatter2 = ax2.scatter(df['r'], df['eta'], c=df[metric2], cmap='viridis', alpha=0.6, edgecolor='none')
    plt.grid(True)
    ax2.set_xlabel('r')
    ax2.set_ylabel('eta')
    ax2.set_title(f'(r, eta) pairs colored by {metric2}')
    cbar2 = fig.colorbar(scatter2, ax=ax2)
    cbar2.set_label(metric2)
    

    plt.subplots_adjust(wspace=0.3)
    plt.show()

    return fig

def create_scatter_plots_log_eta(df, metric=None):
    """
    Create two scatter plots side by side, where the color of each point represents the value from the specified metric column.
    
    Arguments:
    df -- A pandas DataFrame containing the columns 'r', 'eta', and the specified metric column.
    metric -- The name of the column in the DataFrame to use for color mapping.
    """
    if metric:
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1 (Linear eta)
        scatter1 = ax1.scatter(df['r'], df['eta'], c=df[metric], cmap='viridis', alpha=0.8, edgecolor='none')
        ax1.set_xlabel('r')
        ax1.set_ylabel('eta')
        ax1.set_title('(r, eta) pairs colored by {}'.format(metric))

        cbar1 = fig.colorbar(scatter1, ax=ax1)
        cbar1.set_label(metric)
        
        # Plot 2 (Geometric eta)
        mask = df['eta'].isin(10**np.arange(-9.0, 0))
        scatter2 = ax2.scatter(df[mask]['r'], np.log10(df[mask]['eta']), c=df[mask][metric], cmap='viridis', alpha=0.8, edgecolor='none')
        ax2.set_xlabel('r')
        ax2.set_ylabel('eta')
        ax2.set_title('(r, eta) pairs colored by {} (Geometric eta)'.format(metric))

        cbar2 = fig.colorbar(scatter2, ax=ax2)
        cbar2.set_label(metric)

        plt.subplots_adjust(wspace=0.3)

        plt.show()

    else:
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot 1 (Linear eta)
        sns.scatterplot(x=df['r'], y=df['eta'], color='xkcd:shamrock green', alpha=0.5, ax=ax1)
        ax1.set_xlabel('r')
        ax1.set_ylabel('eta')
        ax1.set_title('(r, eta) pairs for which CDFs are computed (Linear eta)')

        # Plot 2 (Geometric eta)
        mask = df['eta'].isin(10**np.arange(-9.0, 0))
        sns.scatterplot(x=df['r'], y=np.log10(df[mask]['eta']), color='xkcd:shamrock green', alpha=0.5, ax=ax2, edgecolor='none')
        ax2.set_xlabel('r')
        ax2.set_ylabel(f'log10 eta')
        ax2.set_title('(r, eta) pairs for which CDFs are computed (Geometric eta)')

        plt.subplots_adjust(wspace=0.3)
        plt.show()

    return fig

def create_contour_plot(df, metric):
    """
    Create a contour plot with a semi-transparent heatmap in the background, where the color represents the values from the specified metric column.
    
    Arguments:
    df -- A pandas DataFrame containing the columns 'r', 'eta', and the specified metric column.
    metric -- The name of the column in the DataFrame to use for color mapping.
    """

    # Create a meshgrid from r and eta
    r_meshgrid, eta_meshgrid = np.meshgrid(df['r'].unique(), df['eta'].unique())
    
    metric_meshgrid = np.zeros_like(r_meshgrid)
    for i, r in enumerate(df['r'].unique()):
        for j, eta in enumerate(df['eta'].unique()):
            mask = (df['r'] == r) & (df['eta'] == eta)
            if mask.any():
                metric_meshgrid[j, i] = df.loc[mask, metric].values[0]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    heatmap = ax.imshow(metric_meshgrid, extent=[df['r'].min(), df['r'].max(), df['eta'].min(), df['eta'].max()], origin='lower', cmap='viridis', alpha=0.5)
    
    contour = ax.contour(r_meshgrid, eta_meshgrid, metric_meshgrid, cmap='viridis')
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label(metric)
    
    ax.clabel(contour, inline=True, fontsize=10)
    ax.set_xlabel('r')
    ax.set_ylabel('eta')
    ax.set_title('Contour Plot (r, eta) colored by {}'.format(metric))
    
    plt.show()

    return fig

def create_ci_scatter_plot(group_cdf_df, values_dict, metric='variance', group=None, ci_levels = [50, 80, 95, 99]):

    thresholds = [np.percentile(values_dict[group], [(100 - ci)/2, ci/2]) for ci in ci_levels]

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=len(ci_levels))

    scatter_collection = []
    for i, (lower, upper) in enumerate(reversed(thresholds)):
        mask = (group_cdf_df[metric] >= lower) & (group_cdf_df[metric] <= upper)
        color = cmap(norm(i))
        scatter = ax.scatter(group_cdf_df.loc[mask, 'r'], group_cdf_df.loc[mask, 'eta'], 
                   c=[color], s=20, alpha=0.7, label=f'{ci_levels[-(i+1)]}% CI')
        scatter_collection.append(scatter)

    ax.set_xlabel('r')
    ax.set_ylabel('eta')
    ax.set_title(f'{metric.capitalize()} Scatter Plot for {group}')

    plt.legend()
    plt.show()
    return fig

def visualize_cdf(params, sample = [], distro='gengamma', n_samples=1000, interval=None, provided_loc=None, all_cdfs=None, group=None):
    """
    Visualize the gap between the empirical CDF and the computed CDF.

    Args:
        sample (np.ndarray): Observed data.
        params (tuple): Parameters for the computed CDF.
        distro (str): Distribution to use for the computed CDF ('gengamma', 'gaussian', or 'laplace').
        n_samples (int): Number of samples for the computed CDF.
        interval (tuple): Optional interval for the x-axis limits.
        provided_loc (float): Optional location to compute the deviation at.
        all_cdfs (dict): Dictionary containing computed CDFs.
        group (int or None): Group index (for titling purposes).

    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
    if len(sample) > 0:
        xs = np.linspace(max(np.min(sample), -100000), min(np.max(sample), 100000), 200000)
        sample = np.sort(sample)
        n = len(sample)

    if distro == 'gengamma':
        if len(params) == 3:
            r, eta, scale = params
        else:
            r, eta = params
            scale = 1
        null_cdf = compute_prior_cdf(r=r, eta=eta, scale = scale, n_samples=n_samples, debug=False)
    elif distro == 'gaussian' or distro == 'normal':
        null_cdf = stats.norm(scale=params).cdf
    elif distro == 'laplace':
        null_cdf = stats.laplace(scale=params).cdf
    elif distro == 't':
        null_cdf = stats.t(df=params[0], scale = params[1]).cdf

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlim(left=-25, right=25)
    if interval:
        ax.set_xlim(left=interval[0], right=interval[1])

    if len(sample) > 0:
        ax.plot(sample, np.arange(1, n+1)/n, label='Empirical CDF')
        result = stats.ks_1samp(sample, null_cdf)
        distance = result.statistic
        location = result.statistic_location
        emp_cdf_at_loc = np.searchsorted(sample, location, side='right') / n
        computed_cdf_at_loc = null_cdf(location)

        ax.vlines(location, emp_cdf_at_loc, computed_cdf_at_loc, linestyles='--',
                label=f'Maximum Deviation: {np.round(distance, 6)}\nat x={np.round(location, 6)}',
                color='xkcd:bright red')

        if provided_loc is not None:
            emp_cdf_at_provided_loc = np.searchsorted(sample, provided_loc, side='right') / n
            computed_cdf_at_provided_loc = null_cdf(provided_loc)
            ax.vlines(provided_loc, emp_cdf_at_provided_loc, computed_cdf_at_provided_loc, linestyles='--',
                    label=f'Deviation: {np.round(np.abs(emp_cdf_at_provided_loc - computed_cdf_at_provided_loc), 6)}\nat x={np.round(provided_loc, 6)}',
                    color='xkcd:shamrock green')
    ax.plot(xs, null_cdf(xs), label='Computed CDF')
    
    if len(sample) > 0:
        if distro == 'gengamma':
            r, eta = params
            ax.set_title(f'{f"{GROUP_NAME} {group}" if group else ""} Empirical CDF vs Computed CDF \n (r={r}, eta={eta}) with p-value:{np.round(result.pvalue, 8)}')
        else:
            ax.set_title(f'{f"{GROUP_NAME} {group}" if group else ""} Empirical CDF vs Computed CDF \n {distro} (0, {params})')
    else:
        ax.set_title(f'Visualized {distro} with params {params}')


    ax.legend()
    plt.tight_layout()

    return fig



def visualize_cdf_pdf(params, sample=[], distro = 'gengamma', log_scale = True, n_samples=2000, interval = None, provided_loc = None, group=None, percent_excluded=0.1, plot_hist=True, bw = 0.05, bw_log = 0.05, binwidth = None):
    """
    Visualize the gap between the empirical CDF/PDF and the Computed CDF/PDF.

    Args:
        sample (np.ndarray): Observed data.
        r (float): r value.
        eta (float): eta value.
        n_samples (int): Number of samples for the computed CDF/PDF.
        all_cdfs (dict): Dictionary containing computed CDFs.
        group (int or None): Group index (for titling purposes).

    Returns:
        distance (float): The Kolmogorov-Smirnov statistic.
        location (float): The location of the maximum deviation between the empirical and computed CDFs.
    """
    if len(sample) > 0:
        
        lower_bound = np.percentile(sample, percent_excluded/2)
        upper_bound = np.percentile(sample, (100-percent_excluded/2))
        original_sample = sample
        sample = sample[(sample > lower_bound) & (sample < upper_bound)]
        sample = np.sort(sample)
        n = len(sample)
        # If interval not specified, set to include 99% of the data
        if interval is None:
            interval = (np.percentile(sample, 5), np.percentile(sample, 95))
        xs = np.linspace(max(interval[0], np.min(sample)), min(np.max(sample), interval[1]), 2000000)
    
    if distro == 'gengamma':
        if len(params) == 3:
            r, eta, scale = params
        else:
            r, eta = params
            scale = 1
        xs_pdf, null_cdf = compute_prior_cdf(r=r, eta=eta, scale = scale, n_samples=n_samples, enforce_assert=False, debug=False, return_xs=True)
        null_pdf = null_cdf.derivative()(xs_pdf)
        
    elif distro == 'gaussian' or distro == 'normal':
        null_cdf = stats.norm(scale=params).cdf
        xs_pdf = np.linspace(-30, 30, 10000)
        null_pdf = stats.norm(scale=params).pdf(xs)
    elif distro == 'laplace':
        null_cdf = stats.laplace(scale=params).cdf
        xs_pdf = np.linspace(-30, 30, 10000)
        null_pdf = stats.laplace(scale=params).pdf(xs_pdf)

    if log_scale:

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        fig.suptitle(f'{f"{GROUP_NAME} {group}" if group else ""} Empirical CDF vs Computed CDF \n (r={r}, eta={eta}, scale={np.format_float_scientific(scale, 3)})', fontsize=14)
        if interval:
            ax1.set_xlim(left = interval[0], right = interval[1])

        if len(sample) > 0:
            ax1.plot(sample, np.arange(1, n+1)/n, label='Empirical CDF')
            result = stats.ks_1samp(original_sample, null_cdf)
            distance = result.statistic
            location = result.statistic_location
            emp_cdf_at_loc = np.searchsorted(original_sample, location, side='right') / n
            computed_cdf_at_loc = null_cdf(location)
            ax1.plot(xs, null_cdf(xs), label='Computed CDF')
            ax1.vlines(location, emp_cdf_at_loc, computed_cdf_at_loc, linestyles='--', label=f'Maximum Deviation: {np.round(distance, 6)}\nat x={np.format_float_scientific(location, 3)}', color='xkcd:bright red')
        else:
            ax1.plot(xs_pdf, null_cdf(xs_pdf), label='Computed CDF')

        if len(sample) > 0 and provided_loc:
            emp_cdf_at_provided_loc = np.searchsorted(original_sample, provided_loc, side='right') / n
            computed_cdf_at_provided_loc = null_cdf(provided_loc)
            ax1.vlines(provided_loc, emp_cdf_at_provided_loc, computed_cdf_at_provided_loc, linestyles='--', label=f'Deviation: {np.round(emp_cdf_at_provided_loc - computed_cdf_at_provided_loc, 6)}\nat x={np.round(provided_loc, 6)}', color='xkcd:shamrock green')

        if interval:
            ax2.set_xlim(left = interval[0], right = interval[1])
        
        if len(sample)>0:
            sns.kdeplot(sample[(sample >= interval[0]) & (sample <= interval[1])], bw_method = bw, ax=ax2, label=f'Empirical PDF (KDE, bw={bw})')
            if plot_hist:
                sns.histplot(sample, ax=ax2, binwidth = binwidth, stat='density', label=f'Empirical PDF ({100-percent_excluded}% of sample)', alpha=0.2)
        ax2.plot(xs_pdf, null_pdf, label='Computed PDF')
       
        if interval:
            ax3.set_xlim(left = interval[0], right = interval[1])
        ax3.set_ylim(bottom = 10**-4, top=10)
        
        if len(sample)>0:
            sns.kdeplot(ax = ax3, x = sample, bw_method = bw_log, log_scale=[False, True], label = f"Empirical PDF (KDE, bw={bw_log})")
            if plot_hist:
                sns.histplot(sample, ax = ax3, binwidth = binwidth, stat = "density", log=True, bins=1000, alpha=0.2, color='#1f77b4', label=f'Empirical PDF ({100-percent_excluded}% of sample)')

        ax3.plot(xs_pdf, null_pdf, label = "Computed PDF")
        
        if len(sample) == 0:
            ax1.set_title(f'Visualized {distro} CDF with params {params}')
            ax2.set_title(f'Visualized {distro} PDF with params {params}')
            ax3.set_title(f'Visualized {distro} PDF (log-scale) with params {params}')
        elif distro == 'gengamma':
            ax1.set_title(f'Empirical CDF vs Computed CDF (p-value:{np.format_float_scientific(result.pvalue, 4)})')
            ax2.set_title(f'Empirical PDF vs Computed PDF')
            ax3.set_title(f'Log Scale: Empirical PDF vs Computed PDF')
        ax1.legend()
        ax2.legend()
        ax3.legend()

        plt.tight_layout()
        plt.show()
        
    return fig

def twoSampleComparisonPlots(samp1, samp2, bw =0.2, samp1name = "Sample 1", samp2name = "Sample 2", alpha = 0.2, plot_hist = True):
    n_1 = len(samp1)
    n_2 = len(samp2)
    ksres = stats.ks_2samp(samp1, samp2)
    ks_loc, ks_stat = ksres.statistic_location, ksres.statistic
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    #axes[0].set_xlim(left = -.25*bound, right = .25*bound)
    #axes[1].set_xlim(left = -.25*bound, right = .25*bound)
    axes[1].set_ylim(bottom = 10**-6, top= 10)
    #axes[2].set_xlim(left = -.25*bound, right = .25*bound)
    sns.kdeplot(ax = axes[0], x = samp1, bw_method=bw, label = samp1name)
    sns.kdeplot(ax = axes[0], x = samp2,bw_method = bw, label = samp2name)
    sns.kdeplot(ax = axes[1], x = samp1, bw_method = bw, log_scale=[False, True], label = samp1name)
    sns.kdeplot(ax = axes[1], x = samp2, bw_method = bw, log_scale=[False, True], label = samp2name)
    if plot_hist:
        sns.histplot(ax = axes[0], x = samp1, stat='density', label = samp1name, alpha=alpha)
        sns.histplot(ax = axes[0], x = samp2, stat='density', label = samp2name, alpha=alpha)
        sns.histplot(ax = axes[1], x = samp1, log = True, stat='density', label = samp1name, alpha=alpha)
        sns.histplot(ax = axes[1], x = samp2, log = True, stat='density', label = samp2name, alpha=alpha)
    axes[2].plot(np.sort(samp1), np.arange(1, n_1+1)/n_1, label=samp1name)
    axes[2].plot(np.sort(samp2), np.arange(1, n_2+1)/n_2, label=samp2name)
    emp_cdf_at_loc = np.searchsorted(np.sort(samp1), ks_loc, side='right') / n_1
    emp_cdf_at_loc2 = np.searchsorted(np.sort(samp2), ks_loc, side='right') / n_2
    axes[2].vlines(ks_loc, emp_cdf_at_loc, emp_cdf_at_loc2, linestyles='--', label=f'Maximum Deviation: {np.round(ks_stat, 6)}\nat x={np.round(ks_loc, 6)}', color='xkcd:bright red')
    
    axes[0].set_title("Non Log Scale Pdf")
    axes[1].set_title("Log Scale Pdf")
    axes[2].set_title(f"CDF with p-value:{np.round(ksres.pvalue, 8)}")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()                                                                
    return fig


def multiSampleComparisonPlots(samps,  samp_names, bw =0.2, alpha = 0.2, hist_plot = True):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    for i in range(len(samps)):
        n_1 = len(samps[i])
        #axes[0].set_xlim(left = -.25*bound, right = .25*bound)
        #axes[1].set_xlim(left = -.25*bound, right = .25*bound)
        axes[1].set_ylim(bottom = 10**-6, top= 10)
        #axes[2].set_xlim(left = -.25*bound, right = .25*bound)
        sns.kdeplot(ax = axes[0], x = samps[i], bw_method=bw, label = samp_names[i])
        sns.kdeplot(ax = axes[1], x = samps[i], bw_method = bw, log_scale=[False, True], label = samp_names[i])

        if hist_plot:
            sns.histplot(ax = axes[0], x = samps[i], stat='density', label = samp_names[i], alpha=alpha)
            sns.histplot(ax = axes[1], x = samps[i], log = True, stat='density', label = samp_names[i], alpha=alpha)

        axes[2].plot(np.sort(samps[i]), np.arange(1, n_1+1)/n_1, label=samp_names[i])

        
        axes[0].set_title("Non Log Scale Pdf")
        axes[1].set_title("Log Scale Pdf")
        axes[2].set_title(f"CDF")
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()                                                                
    return fig



def nearby_df_parallel_process(r_prime, eta_prime, prior_cdf, n, ks_max, iterations, rounded):
    total_distance, total_pvalue = 0, 0
    r_prime = np.round(r_prime, rounded)
    eta_prime = np.round(eta_prime, rounded)

    for _ in range(iterations):
        obs_x = sample_prior(r_prime, eta_prime, size=n)
        filtered_x = np.sort(obs_x)[np.round(np.linspace(0, obs_x.size - 1, min(obs_x.size, ks_max))).astype(int)] 
        distance, _ = kstest_custom(filtered_x, prior_cdf)
        pvalue = 1 - stats.kstwo(n=n).cdf(distance)
        total_distance += distance
        total_pvalue += pvalue
    
    avg_distance = total_distance / iterations
    avg_pvalue = total_pvalue / iterations
    
    return [r_prime, eta_prime, avg_distance, avg_pvalue]

def plotKSHeatMap(df, r, eta, grid_amt= 5, pval =True, dist = True, title = "", plot_fig = True):
    if dist:
        result = df.pivot(index='eta', columns='r', values='distance').sort_values("eta", ascending=True)
        
        if grid_amt != 1:
            fig_dist, ax = plt.subplots(figsize=(1.6*grid_amt, 1.6*grid_amt))
            sns.heatmap(result, annot=True, fmt=f".3f", ax =ax, square=True)
        else:
            fig_dist, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(result, annot=True, fmt=f".3f", ax =ax, square=True, cbar=False)
        
       
        plt.title(f"{title}, r = {r} eta = {eta}, Distances")
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        ax.invert_yaxis()
        if plot_fig:
            plt.show()
    if pval:
        result = df.pivot(index='eta', columns='r', values='pvalue').sort_values("eta", ascending=True)
        if grid_amt != 1:
            fig_pval, ax = plt.subplots(figsize=(1.6*grid_amt, 1.6*grid_amt))
            sns.heatmap(result, annot=True, fmt=f".3f", cmap = "RdYlGn", vmin = 0.01, vmax = 0.2, square=True)
        else:
            fig_pval, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(result, annot=True, fmt=f".3f", cmap = "RdYlGn", vmin = 0.01, vmax = 0.2, square=True, cbar=False)
        
        
        plt.title(f"{title}, r = {r} eta = {eta}, pvalues")
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        ax.invert_yaxis()
        if plot_fig:
            plt.show()
    return [fig_dist, fig_pval]

def KSHeatMap(r, eta, n=10000, ks_max = 100000, r_bound=0.01, eta_bound =0.1, grid_amt= 5, iterations = 10, rounded = 3, pval =True, dist = True, title = ""):
    df = nearby_df(r, eta, iterations=iterations, n=n, ks_max = ks_max, r_bound = r_bound, eta_bound = eta_bound, grid_amt=grid_amt, rounded=rounded)
    plotKSHeatMap(df=df, r=r, eta=eta, grid_amt=grid_amt, pval=pval, dist =  dist, title = title)

def center_square_check(r, eta, n=10000, ks_max=100000, prior_cdf = None, iterations=10):
    df = pd.DataFrame(columns = ["r", "eta", "distance", "pvalue"])
    if prior_cdf == None:
        prior_cdf = compute_prior_cdf(r, eta, n_samples=1000, tail_percent=0.1, tail_bound=0.0001, debug=False, use_matlab=True, eng=eng)
    total_distance, total_pvalue = 0, 0
    for _ in range(iterations):
        obs_x = sample_prior(r, eta, size = n)
        filtered_x = np.sort(obs_x)[np.round(np.linspace(0, obs_x.size - 1, min(obs_x.size, ks_max))).astype(int)] 
        distance, _ = kstest_custom(filtered_x, prior_cdf)
        pvalue = 1 - stats.kstwo(n=n).cdf(distance)
        total_distance += distance
        total_pvalue += pvalue
    
    avg_distance = total_distance/iterations
    avg_pvalue = total_pvalue/iterations
    df.loc[len(df)] = [r, eta, avg_distance, avg_pvalue]

    return df

def KSHeatMapFullProcess(r, eta, n=10000, ks_max = 100000, r_bound=0.01, eta_bound =0.1, grid_amt= 5, iterations = 10, dist = True, pval = True, rounded = 4, accept_pval = 0.05, good_pct = 0.8, title= "", return_vals = False, print_messages = True, max_iterations = 6, parallelize = True):
    
    bound_divide = 10
    prior_cdf = compute_prior_cdf(r, eta, n_samples=1000, tail_percent=0.1, tail_bound=0.0001, debug=False, use_matlab=True, eng=eng)
    if print_messages:
        print("Check Center Square")
    center_test = center_square_check(r = r, eta = eta, n = n, ks_max = ks_max, iterations = iterations, prior_cdf = prior_cdf)
    if center_test["pvalue"][0] < accept_pval:
        if print_messages:
            print("Center Square Failed No Need to test the rest")
        print(center_test)
        failed_figs = plotKSHeatMap(df=center_test, r = r, eta = eta, grid_amt = 1, pval=True, dist = True, title = title + " Failed Center Square")
        return [failed_figs, failed_figs], [0, 0, 0, None, None, None]
    else:
        if print_messages:
            print("Center Square Passed")

    df = nearby_df(r=r, eta=eta, n=n, ks_max = ks_max, r_bound=r_bound, eta_bound=eta_bound, grid_amt = grid_amt, iterations= iterations, rounded=rounded, parallelize = parallelize, prior_cdf = prior_cdf)
    intial_fig = plotKSHeatMap(df=df, r=r, eta= eta, grid_amt = grid_amt, pval=pval, dist = dist, title = title + " Original Bounds")
    pass_pct = len(df[df["pvalue"] >= accept_pval])/len(df)
    initial_pct = pass_pct
    initial_r_bound = r_bound
    initial_eta_bound = eta_bound
    if print_messages:
        print("Running process with original bounds")
    if pass_pct < good_pct:
        if print_messages:
            print(f"Only {pass_pct*100}% of tests passed with the original bounds. Now running with lower r and eta bounds")
        while pass_pct < good_pct and max_iterations>0:
            r_bound = r_bound/bound_divide
            eta_bound = eta_bound/bound_divide
            if bound_divide == 2:
                bound_divide = 5
            elif bound_divide == 5:
                bound_divide = 2
            if print_messages:
                print(f"Trying r_bound = {r_bound}, eta_bound = {r_bound}")
            df = nearby_df(r=r, eta=eta, n=n, ks_max = ks_max, r_bound=r_bound, eta_bound=eta_bound, grid_amt = grid_amt, iterations= iterations, rounded=rounded)
            pass_pct = len(df[df["pvalue"] >= accept_pval])/len(df)
            if pass_pct < good_pct:
                if print_messages:
                    print(f"Only {pass_pct*100}% of tests passed using r_bound = {r_bound}, eta_bound = {eta_bound}.Now running with lower r and eta bounds")
            else:
                if print_messages:
                    print(f"{pass_pct*100}% of tests passed using r_bound = {r_bound}, eta_bound = {eta_bound}. Showing Heatmaps")
                
            max_iterations -= 1
        final_fig = plotKSHeatMap(df=df, r=r, eta= eta, grid_amt = grid_amt, pval=pval, dist = dist, title = title+ " Final Bounds")
    else:
        if print_messages:
            print(f"{pass_pct*100}% of tests passed with the original bounds.")
    if return_vals:
        return [intial_fig, final_fig], [initial_r_bound, initial_eta_bound, initial_pct, r_bound, eta_bound, pass_pct]
    





def create_region_scatter_plot(df,  x_col = "r", y_col = "eta", metric=None, plot_name = '', log_colorbar=False):
    """
    Create a scatter plot, where the color of each point represents the value from the specified metric column.
    If metric=None, plot all the (r, eta) values in df.

    Arguments:
    df : A pandas DataFrame containing the columns 'r', 'eta', and the specified metric column.
    metric : The name of the column in the DataFrame to use for color mapping.
    log_scale : Boolean, if True, the color scale of the plot will be logarithmic.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    if metric:
        if pd.api.types.is_numeric_dtype(df[metric]):
            norm = LogNorm() if log_colorbar else None
            scatter = ax.scatter(df[x_col], df[y_col], c=df[metric], cmap='viridis', alpha=1, norm=norm)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(metric)
        else:
            categories = df[metric].unique()
            color_map = plt.colormaps['accent']
            colors = {cat: color_map(i/len(categories)) for i, cat in enumerate(categories)}
            
            for cat in categories:
                mask = df[metric] == cat
                ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col], 
                           c=[colors[cat]], label=cat, alpha=1)
            
            ax.legend(title=metric)
        if plot_name:
            ax.set_title(plot_name)
        else:
            ax.set_title(f'({x_col}, {y_col}) pairs colored by {metric}')
    else:
        sns.scatterplot(x=df[x_col], y=df[y_col], color='xkcd:gray', alpha=0.5, ax=ax, edgecolor='none')
        ax.set_title(f'({x_col}, {y_col}) pairs for which CDFs are computed (Linear eta)')

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.grid(which='both')
    plt.show()

    return fig




def region_reporting(rEtaKsstats_dict, master_df, layer = "all", MULT =1.2, kind = "Layer", plots=True, plot_name = ''):
    if layer == "all":
        layers = master_df.index
    else:
        layers = layer
    hulls_df = pd.DataFrame(columns=["BAND", "hull"])
    for BAND in layers:

        
        drop_keys =list(rEtaKsstats_dict[BAND].keys())[-3:]
        print(drop_keys)
        test = pd.DataFrame(rEtaKsstats_dict[BAND]).drop(drop_keys, axis = 1 )
        test4 = pd.DataFrame(rEtaKsstats_dict[BAND])[drop_keys]
        test4 = test4.rename(columns = {"r_optimize": "r", "eta_optimize": "eta", drop_keys[-1]: "ksstat"})
        test4 = test4.dropna()
        test = test.merge(test4, on=["r", "eta"], how="outer")
        test = test.set_index(["r", "eta"])
        test["ksstat"] = test.min(axis=1)
        test = test.reset_index()
        test = test[["r", "eta", "ksstat"]]
        print(min(test["ksstat"]) * MULT, master_df.loc[BAND, "kstest_stat_cutoff_0.05"])
        
        cutoff = max(min(test["ksstat"]) * MULT, master_df.loc[BAND, "kstest_stat_cutoff_0.05"], 0.01)
        if cutoff != master_df.loc[BAND, "kstest_stat_cutoff_0.05"]:
            print(f"Layer {BAND}: None Passed")
        test["map"] =  test["ksstat"].apply(lambda x: "pass" if x <= cutoff else "")
        test["1/beta"] = test["r"]/(test["eta"] + 1.5)
        test["log_r"] = np.log10(test["r"])
        test["log_beta"] = np.log10((test["eta"] + 1.5)/ test["r"])
        test["beta"] = (test["eta"] + 1.5) / test["r"]
        test["1/log_beta"] = 1/np.log10((test["eta"] + 1.5)/ test["r"])

    

        test2 = test[test["ksstat"] <= cutoff]
        
        if plots:

            #create_region_scatter_plot(test, x_col = "r", y_col = "eta", metric='ksstat', plot_name=f"{plot_name} KS Stats + {kind}: {BAND} eta space", log_colorbar=True)
            create_region_scatter_plot(test2, x_col = "r", y_col = "eta", metric='ksstat', plot_name=f"{plot_name} KS Stats + {kind}: {BAND} eta space", log_colorbar=True)
            #create_region_scatter_plot(test2, x_col = "r", y_col = "beta", metric='ksstat', plot_name=f"{plot_name} KS Stats + {kind}: {BAND} eta space", log_colorbar=True)
            #create_region_scatter_plot(test2, x_col = "log_r", y_col = "log_beta", metric='ksstat', plot_name=f"{plot_name} KS Stats + {kind}: {BAND} log log space", log_colorbar=True)
            #create_region_scatter_plot(test2, x_col = "r", y_col = "1/log_beta", metric='ksstat', plot_name=f"{plot_name} KS Stats + {kind}: {BAND} log log space", log_colorbar=True) 
            #create_region_scatter_plot(test2, x_col = "r", y_col = "log_beta", metric='ksstat', plot_name=f"{plot_name} KS Stats + {kind}: {BAND} log beta space", log_colorbar=True) 
            #create_region_scatter_plot(test, x_col = "r", y_col = "1/beta", metric='ksstat', plot_name=f"{plot_name} KS Stats + {kind}: {BAND} 1/beta space", log_colorbar=True) 
            create_region_scatter_plot(test2, x_col = "r", y_col = "1/beta", metric='ksstat', plot_name=f"{plot_name} KS Stats + {kind}: {BAND} 1/beta space", log_colorbar=True) 


        # Define a rational function: f(r) = (a * r^2 + b * r + c) / (d * r + e)

        if len(test2) > 5:
            def rational_func(r, a, b, c, d, e):
                return (a * r**2 + b * r + c) / (d * r + e) 

            def quadratic_func(r, a, b, c):
                return a * r**2 + b * r + c

            def linear_func(r, m, c):
                return m * r + c

            # Fit the rational function to the data
            popt_rational, _ = curve_fit(rational_func, test2["r"], test2["1/beta"], maxfev=10000)

            # Fit the quadratic function to the data
            popt_quadratic, _ = curve_fit(quadratic_func, test2["r"], test2["1/beta"], maxfev=10000)

            # Fit the linear function to the data
            popt_linear, _ = curve_fit(linear_func, test2["r"], test2["1/beta"], maxfev=10000)

            # Generate predictions using the fitted parameters
            r_values = np.linspace(test2["r"].min(), test2["r"].max(), 500)
            fitted_rational = rational_func(r_values, *popt_rational)
            fitted_quadratic = quadratic_func(r_values, *popt_quadratic)
            fitted_linear = linear_func(r_values, *popt_linear)

            # Extract the fitted parameters
            a, b, c, d, e = popt_rational
            print(f"Rational Function Parameters: a={a}, b={b}, c={c}, d={d}, e={e}")
            a_q, b_q, c_q = popt_quadratic
            print(f"Quadratic Function Parameters: a={a_q}, b={b_q}, c={c_q}")
            m, c_l = popt_linear
            print(f"Linear Function Parameters: m={m}, c={c_l}")

            points = np.column_stack((test2["r"], test2["1/beta"])) + stats.norm.rvs(size=(len(test2), 2)) * 0.001  # Adding small noise for convex hull computation
            # Compute the convex hull
            hull = ConvexHull(points)
            hull_coordinates = points[hull.vertices]
            hulls_df = pd.concat([hulls_df, pd.DataFrame({"BAND": [BAND], "hull": [hull_coordinates]})], ignore_index=True)
            
            if plots:
                plt.figure(figsize=(8, 6))
                plt.scatter(test2["r"], test2["1/beta"], label="Data", color="blue", alpha=0.6)
                plt.plot(r_values, fitted_rational, label="Rational Function", color="red", linewidth=2)
                plt.plot(r_values, fitted_quadratic, label="Quadratic Function", color="green", linestyle="--", linewidth=2)
                plt.plot(r_values, fitted_linear, label="Linear Function", color="orange", linestyle=":", linewidth=2)

                
                # Plot the convex hull
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], color = "black")
                    
                plt.xlabel("r")
                plt.ylabel("1/beta")
                plt.title(f"{plot_name} {kind}: {BAND} Fitted Functions")
                plt.legend()
                plt.grid()
                plt.show()
    return hulls_df

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
            hull = ConvexHull(points)
            master_df_copy.loc[group, "hull"] = hull

    return master_df_copy.reset_index()

   


def plot_pdf_components(r, eta, scale, components = 10, mode = "equal", color_fn = None, custom_var = None, varlim = None, xlim = None, lin_lim = None, log_lim = None, debug = True, prior_pdf = None, title = None, edgecolor = 'black'):
    beta = (eta + 1.5) / r
    
    if mode == "equal":
        x = np.linspace(0, 1, components+2)[1:-1][::-1]  # Exclude 0 and 1 to avoid singularities
        vars = stats.gengamma(a=beta, c=r, scale=scale).ppf(x)
        weights = np.ones(components) / components
    elif mode == "variance":
        if custom_var is None and varlim is None:
            raise ValueError("Either custom_var or varlim must be provided for 'variance' mode.")
        if custom_var is not None:
            vars = np.array(custom_var)
        elif varlim is not None:
            if varlim[2] == "linear":
                vars = np.linspace(varlim[0], varlim[1], components)[::-1]
            elif varlim[2] == "log":
                vars = np.logspace(np.log10(varlim[0]), np.log10(varlim[1]), components)[::-1]
        weights = stats.gengamma(a = beta, c = r, scale=scale).cdf(vars[::-1])
        weights = np.diff(weights, prepend=0)[::-1]
        print(sum(weights))  # Convert cumulative weights to densities
        print(weights)
          

        
    
    if prior_pdf is None:
        xs, genGamma_prior = compute_prior_pdf(r=r, eta=eta, scale=scale, n_samples=2000, debug=debug)
    else:
        xs = np.linspace(-5, 5, 1000)
        genGamma_prior = prior_pdf

    means = np.zeros(components)
    print(vars)
    norm_pdfs = np.array([stats.norm.pdf(xs, loc=mean, scale=np.sqrt(var)) for mean, var in zip(means, vars)])
    if color_fn is None:
        cmap = plt.cm.viridis(np.linspace(0, 1, components))
    else:
        cmap = [color_fn(var) for var in vars]  # Create a colormap based on the variances

    # Create side-by-side subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True, constrained_layout=True)

    for ax, yscale in zip(axs, ['linear', 'log']):
    # Plot stackplot
        ax.stackplot(
            xs,
            *(norm_pdfs * weights[:, None]),
            colors=cmap,
            labels=[f'Component {i+1} (var={vars[i]:.2f}), (wt={weights[i]:.2f})' for i in range(components)]
        )
        # Overlay the GenGamma prior
        ax.plot(xs, genGamma_prior(xs), label='GenGamma Prior', color='black', linestyle='--', linewidth=2)

        # Set styles
        for polygon in ax.collections:
            polygon.set_edgecolor(edgecolor)
        ax.set_xlabel('x')
        ax.set_ylabel('Density')

        ax.set_yscale(yscale)


    # Shared title and colorbar
    if xlim is not None:
        axs[0].set_xlim(xlim)
        axs[1].set_xlim(xlim)
    if lin_lim is not None:
        axs[0].set_ylim(lin_lim)
    if log_lim is not None:
        axs[1].set_ylim(log_lim)

    if title is None:
        title = f'Scale Mixture of Normals with GenGamma Prior\nr={r}, eta={eta}, scale={scale}, components={components}'
    fig.suptitle(title, fontsize=14)


    # Add colorbar outside the plots
    #sm = plt.cm.ScalarMappable(
        #cmap=plt.cm.viridis_r,  # reversed colormap
        #norm=colors.Normalize(vmin=vars.min(), vmax=vars.max())
    #)  
    #sm.set_array([])
    #cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), location='right', pad=0.02, aspect=40)
    #cbar.set_label('Variance')

    # Only show legend for the first axis, outside the plot
    handles, labels = axs[0].get_legend_handles_labels()
    # Remove duplicate labels (e.g., GenGamma Prior)
    unique = dict(zip(labels, handles))
    #fig.legend(unique.values(), unique.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, borderaxespad=0.)
    return fig
    