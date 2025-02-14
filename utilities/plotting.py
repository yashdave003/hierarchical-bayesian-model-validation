import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
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
        sns.scatterplot(x = 0, y = 0, marker='*', s = 1, c = 'xkcd:electric pink', ax=ax, label = f'KS Stat: {np.round(best_ksstat, decimals=3)}', edgecolor='none')
    plt.legend(loc = 'lower right')
    if plot_name:
        plt.title(plot_name)
    else:
        plt.title(f"{', '.join([col[5:].capitalize() for col in cols])} with boundary {extra_boundary}")

   
    plt.show()
    return fig

def create_scatter_plot(df, metric=None, log_scale=False):
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
            norm = LogNorm() if log_scale else None
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

def visualize_cdf_pdf(params, sample=[], distro = 'gengamma', log_scale = True, n_samples=1000, interval = None, provided_loc = None, group=None, bw = 0.05, bw_log = 0.05):
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
        xs = np.linspace(max(-100000, np.min(sample)), min(np.max(sample), 100000), 2000000)
        sample = np.sort(sample)
        n = len(sample)
    
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

        # Empirical CDF vs Computed CDF
        ax1.set_xlim(left = -25, right = 25)
        if interval:
            ax1.set_xlim(left = interval[0], right = interval[1])

        if len(sample) > 0:
            ax1.plot(sample, np.arange(1, n+1)/n, label='Empirical CDF')
            result = stats.ks_1samp(sample, null_cdf)
            distance = result.statistic
            location = result.statistic_location
            emp_cdf_at_loc = np.searchsorted(sample, location, side='right') / n
            computed_cdf_at_loc = null_cdf(location)
            ax1.plot(xs, null_cdf(xs), label='Computed CDF')
            ax1.vlines(location, emp_cdf_at_loc, computed_cdf_at_loc, linestyles='--', label=f'Maximum Deviation: {np.round(distance, 6)}\nat x={np.round(location, 6)}', color='xkcd:bright red')
        else:
            ax1.plot(xs_pdf, null_cdf(xs_pdf), label='Computed CDF')

        if len(sample) > 0 and provided_loc:
            emp_cdf_at_provided_loc = np.searchsorted(sample, provided_loc, side='right') / n
            computed_cdf_at_provided_loc = null_cdf(provided_loc)
            ax1.vlines(provided_loc, emp_cdf_at_provided_loc, computed_cdf_at_provided_loc, linestyles='--', label=f'Deviation: {np.round(emp_cdf_at_provided_loc - computed_cdf_at_provided_loc, 6)}\nat x={np.round(provided_loc, 6)}', color='xkcd:shamrock green')

        # Empirical PDF vs Computed PDF
        ax2.set_xlim(left = -25, right = 25)
        if interval:
            ax2.set_xlim(left = interval[0], right = interval[1])
        
        sns.kdeplot(sample, bw_method = bw, ax=ax2, label=f'Empirical PDF (KDE, bw={bw})')
        ax2.plot(xs_pdf, null_pdf, label='Computed PDF')
        
        # Log Scale
        ax3.set_xlim(left = -25, right = 25)
        if interval:
            ax3.set_xlim(left = interval[0], right = interval[1])
        ax3.set_ylim(bottom = 10**-4, top=10)
        sns.kdeplot(ax = ax3, x = sample, bw_method = bw, log_scale=[False, True], label = f"Empirical PDF (KDE, bw={bw_log})")
        ax3.plot(xs_pdf, null_pdf, label = "Computed PDF")

        if len(sample) == 0:
            ax1.set_title(f'Visualized {distro} CDF with params {params}')
            ax2.set_title(f'Visualized {distro} PDF with params {params}')
            ax3.set_title(f'Visualized {distro} PDF (log-scale) with params {params}')
        elif distro == 'gengamma':
            ax1.set_title(f'{f"{GROUP_NAME} {group}" if group else ""} Empirical CDF vs Computed CDF \n (r={r}, eta={eta}) with p-value:{np.round(result.pvalue, 8)}')
            ax2.set_title(f'{f"{GROUP_NAME} {group}" if group else ""} Empirical PDF vs Computed PDF \n (r={r}, eta={eta})')
            ax3.set_title(f'{f"{GROUP_NAME} {group}" if group else ""} Log Scale:\n Empirical PDF vs Computed PDF (r={r}, eta={eta})')
        else:
            ax1.set_title(f'{f"{GROUP_NAME} {group}" if group else ""} Empirical CDF vs Computed CDF \n {distro} (0, {params})')
            ax2.set_title(f'{f"{GROUP_NAME} {group}" if group else ""} Empirical PDF vs Computed PDF \n {distro} (0, {params})')
            ax3.set_title(f'{f"{GROUP_NAME} {group}" if group else ""} Log Scale:\n Empirical PDF vs Computed PDF {distro} (0, {params})')

        ax1.legend()
        ax2.legend()
        ax3.legend()

        plt.tight_layout()
        plt.show()

    else:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Empirical CDF vs Computed CDF
        ax1.set_xlim(left = -25, right = 25)
        ax1.plot(sample, np.arange(1, n+1)/n, label='Empirical CDF')
        ax1.plot(xs, null_cdf(xs), label='Computed CDF')
        result = stats.ks_1samp(sample, null_cdf)
        distance = result.statistic
        location = result.statistic_location
        emp_cdf_at_loc = np.searchsorted(sample, location, side='right') / n
        computed_cdf_at_loc = null_cdf(location)
        ax1.vlines(location, emp_cdf_at_loc, computed_cdf_at_loc, linestyles='--', label=f'Maximum Deviation: {np.round(distance, 6)}\nat x={np.round(location, 6)}', color='xkcd:bright red')
        if distro =='gengamma':
            ax1.set_title(f'{f"{GROUP_NAME} {group}" if group else ""} Empirical CDF vs Computed CDF \n (r={r}, eta={eta}) with p-value:{np.round(result.pvalue, 8)}')
            ax2.set_title(f'{f"{GROUP_NAME} {group}" if group else ""} Empirical PDF vs Computed PDF \n (r={r}, eta={eta})')
        else:
            ax1.set_title(f'{f"{GROUP_NAME} {group}" if group else ""} Empirical CDF vs Computed CDF \n {distro} (0, {params})')
            ax2.set_title(f'{f"{GROUP_NAME} {group}" if group else ""} Empirical PDF vs Computed PDF \n {distro} (0, {params})')
        ax1.legend()

        # Empirical PDF vs Computed PDF
        ax2.set_xlim(left = -25, right = 25)
        sns.kdeplot(sample, bw_method = bw, ax=ax2, label=f'Empirical PDF (KDE, bw={bw})')
        ax2.plot(xs_pdf, null_pdf, label='Computed PDF')
        ax2.legend()
    
    return fig



def twoSampleComparisonPlots(samp1, samp2, bw =0.2, samp1name = "Sample 1", samp2name = "Sample 2"):
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


def multiSampleComparisonPlots(samps,  samp_names, bw =0.2):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    for i in range(len(samps)):
        n_1 = len(samps[i])
        #axes[0].set_xlim(left = -.25*bound, right = .25*bound)
        #axes[1].set_xlim(left = -.25*bound, right = .25*bound)
        axes[1].set_ylim(bottom = 10**-6, top= 10)
        #axes[2].set_xlim(left = -.25*bound, right = .25*bound)
        sns.kdeplot(ax = axes[0], x = samps[i], bw_method=bw, label = samp_names[i])
        sns.kdeplot(ax = axes[1], x = samps[i], bw_method = bw, log_scale=[False, True], label = samp_names[i])
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