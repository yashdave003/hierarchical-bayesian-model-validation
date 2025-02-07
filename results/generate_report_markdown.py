import os
import pandas as pd
from pathlib import Path
import git
import argparse

def get_project_root():
    return Path(git.Repo('.', search_parent_directories=True).working_tree_dir)

def get_relative_plot_path(plots_path, output_path, filename):
    """Get the relative path for a plot file from the output path."""
    plot_full_path = os.path.abspath(os.path.join(plots_path, filename))
    relative_path = os.path.relpath(plot_full_path, start=output_path)
    return relative_path

def create_image_grid(plots_path, output_path, plot_name, groups, cols=2, plot_type="grid", img_width="45%"):
    rows = -(-len(groups) // cols)
    grid = ""
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(groups):
                group = groups[idx]
                if plot_type == "grid":
                    filename = f"{plot_name}_layer{group}.jpg"
                elif plot_type == "compare":
                    filename = f"compare_cdf_pdf_layer_{group}.jpg"
                grid += f'<img src="{get_relative_plot_path(plots_path, output_path, filename)}" alt="Layer {group} Plot" width="{img_width}"/>\n'
        grid += "\n"
    return grid

def create_comparison_grid(plots_path, output_path, groups, plot_pairs, cols=2, img_width="80%"):
    rows = -(-len(groups) // cols)
    grid = ""
    for i in range(rows):
        grid += '<div style="display: flex; justify-content: space-around; margin-bottom: 20px;">\n'
        for j in range(cols):
            idx = i * cols + j
            if idx < len(groups):
                group = groups[idx]
                grid += f'  <div style="text-align: center;">\n'
                grid += f'    <p style="font-weight: bold;">Layer {group}</p>\n'
                grid += f'    <div style="display: flex; gap: 10px;">\n'
                for filename_template, label in plot_pairs:
                    filename = filename_template.format(group)
                    grid += f'      <div>\n'
                    grid += f'        <img src="{get_relative_plot_path(plots_path, output_path, filename)}" alt="{label} Layer {group}" style="width: {img_width};">\n'
                    grid += f'        <p>{label}</p>\n'
                    grid += f'      </div>\n'
                grid += f'    </div>\n'
                grid += f'  </div>\n'
        grid += '</div>\n\n'
    return grid

def generate_markdown_report(data_name):
    """Generate markdown report for the dataset."""
    # Parse data_name into its components
    size, dataset_name, representation, channel = data_name.split('-')
    
    # Set up paths relative to results directory
    base_path = os.path.join(get_project_root(), "results")
    
    # Construct relative paths using os.path.join
    plots_path = os.path.join("case-studies", dataset_name, representation, size, channel, "plots")
    csv_path = os.path.join("case-studies", dataset_name, representation, size, channel, "CSVs")
    
    output_path = os.path.join(base_path)
    os.makedirs(output_path, exist_ok=True)

    # Read data
    index_col = 'layer' if representation == 'wavelet' else 'band'
    master_df = pd.read_csv(os.path.join(base_path, csv_path, "master_df.csv"), index_col=index_col)
    layers_or_bands = master_df.index.tolist()

    # Define plot pairs
    combo_plot_pairs = [
        ("full_grid_search_combo_plot_layer{}.jpg", "Full Grid Search"),
        ("optimized_full_grid_search_combo_plot_layer{}.jpg", "Fine Grid Search")
    ]

    markdown_content = f"""# {dataset_name.upper()} Dataset ({representation.capitalize()}) - {pd.Timestamp.now().strftime('%Y-%m-%d')}
## Dataset Description
* **Original source:** [Add source information here]
* **Sizes:** [Add size information here]
* **Image Type:** {channel.capitalize()}
* **Date range covered:** [Add date range here]
* **Number of Images (and channels):** [Add number of images here]
* **Representation:** {representation.capitalize()}

## Why did we choose it?
[Add reasons for choosing this dataset]

## Cleaning - what did we do?
[Add cleaning process details]

## Hypotheses
[Add hypotheses, basis/representation used, and assumptions about signal subsets]

## Tests and Questions
### Full Grid Search Combo Plots
{create_comparison_grid(plots_path, output_path, layers_or_bands, combo_plot_pairs, cols=1, img_width="80%")}

### Compare CDF PDF Plots
{create_image_grid(plots_path, output_path, "compare_cdf_pdf", layers_or_bands, cols=1, plot_type="compare", img_width="90%")}

## Results
### Best parameters from the proposed prior distribution:
{master_df[['total_samples', 'best_r', 'best_eta', 'kstest_stat_best', 'kstest_stat_cutoff_0.05', 'n_pval_0.05']].to_markdown()}

### Optimization progression:
{master_df.filter(regex='.*_r$|.*_eta$|iter.*').to_markdown()}

### Parameter comparisons with other common priors (Gaussian, Laplace, Student t):
{master_df.filter(regex='param.*|kstest_stat.*[^0-9]$|kstest_stat_cutoff.*').to_markdown()}

### All the columns you can access:
{list(master_df.columns)}

## Major Take-aways
[Add major conclusions and insights]"""

    output_file = os.path.join(base_path, "draft_reports", f"{data_name}.md")
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    print(f"Markdown report generated: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate markdown report for dataset analysis')
    parser.add_argument('data_name', help='Data name in the format <size>-<dataset_name>-<representation>-<channel>')
    args = parser.parse_args()

    generate_markdown_report(args.data_name)
