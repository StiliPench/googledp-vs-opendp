import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# load the combined results
results_filepath_csv = "combined_laplace_results.csv"
try:
    plot_df_combined = pd.read_csv(results_filepath_csv)
    print(f"Successfully loaded results from {results_filepath_csv}")
except FileNotFoundError:
    print(f"Error: Results file '{results_filepath_csv}' not found.")
    print("Please run the experiment script (e.g., comparison_plotter.py) first to generate and save the results.")
    exit()

if plot_df_combined.empty:
    print("Loaded DataFrame is empty. No data to plot.")
    exit()

# plotting combined results
sns.set_theme(style="whitegrid", context="talk")
plot_output_dir = "comparison_plots"
os.makedirs(plot_output_dir, exist_ok=True)

marker_size = 23
line_width = 5
marker_edge_width = 1.5

# utility per dataset size
for query_name in ['Count', 'Sum']:
    plt.figure(figsize=(15, 9))
    data_to_plot = plot_df_combined[plot_df_combined['query'] == query_name]
    if data_to_plot.empty:
        print(f"No data for {query_name} query to plot MAE vs Epsilon.")
        continue
    sns.lineplot(data=data_to_plot, x='epsilon', y='mae', hue='library', style='dataset_name', 
                 markers=True, dashes=True, 
                 markersize=marker_size, 
                 linewidth=line_width,
                 markeredgewidth=marker_edge_width)
    plt.title(f'Utility (MAE vs. Epsilon) for {query_name} Query - Laplace')
    plt.xlabel('Epsilon (Privacy Budget)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend(title='Legend', fontsize='medium')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, f'utility_vs_epsilon_{query_name.lower()}_laplace.png'))
    plt.show()

# performance per dataset size
for query_name in ['Count', 'Sum']:
    plt.figure(figsize=(15, 9))
    data_to_plot = plot_df_combined[plot_df_combined['query'] == query_name]
    if data_to_plot.empty:
        print(f"No data for {query_name} query to plot Time vs Epsilon.")
        continue
    sns.lineplot(data=data_to_plot, x='epsilon', y='avg_time_ms', hue='library', style='dataset_name', 
                 markers=True, dashes=True, 
                 markersize=marker_size, 
                 linewidth=line_width,
                 markeredgewidth=marker_edge_width)
    plt.title(f'Performance (Time vs. Epsilon) for {query_name} Query - Laplace')
    plt.xlabel('Epsilon (Privacy Budget)')
    plt.ylabel('Average Execution Time (ms)')
    plt.legend(title='Legend', fontsize='medium')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, f'performance_vs_epsilon_{query_name.lower()}_laplace.png'))
    plt.show()

# scalability for key epsilon values
key_epsilons_plot = [0.1, 1.0, 3.0] 
plot_df_key_eps_combined = plot_df_combined[plot_df_combined['epsilon'].isin(key_epsilons_plot)]

unique_dataset_sizes_sorted = sorted(plot_df_combined['dataset_size'].unique())
dataset_size_tick_labels = []
for size_val in unique_dataset_sizes_sorted:
    if size_val == 10000: dataset_size_tick_labels.append('Small (10k)')
    elif size_val == 100000: dataset_size_tick_labels.append('Medium (100k)')
    elif size_val == 1000000: dataset_size_tick_labels.append('Large (1M)')
    else: dataset_size_tick_labels.append(str(size_val))

for query_name in ['Count', 'Sum']:
    plt.figure(figsize=(15, 9))
    data_to_plot = plot_df_key_eps_combined[plot_df_key_eps_combined['query'] == query_name]
    if data_to_plot.empty:
        print(f"No data for {query_name} query to plot MAE vs Dataset Size (key epsilons).")
        continue
    sns.lineplot(data=data_to_plot, x='dataset_size', y='mae', hue='library', style='epsilon', 
                 markers=True, palette='viridis', dashes=True, 
                 markersize=marker_size, 
                 linewidth=line_width,
                 markeredgewidth=marker_edge_width)
    plt.title(f'Scalability of Utility (MAE vs. Dataset Size) for {query_name} Query - Laplace')
    plt.xlabel('Dataset Size (Number of Rows)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.xscale('log')
    if unique_dataset_sizes_sorted:
        plt.xticks(unique_dataset_sizes_sorted, dataset_size_tick_labels, rotation=30, ha='right')
    plt.legend(title='Legend', loc='best', fontsize='medium')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, f'scalability_utility_vs_datasize_{query_name.lower()}_laplace.png'))
    plt.show()

# scalability for key epsilon values
for query_name in ['Count', 'Sum']:
    plt.figure(figsize=(15, 9))
    data_to_plot = plot_df_key_eps_combined[plot_df_key_eps_combined['query'] == query_name]
    if data_to_plot.empty:
        print(f"No data for {query_name} query to plot Time vs Dataset Size (key epsilons).")
        continue
    sns.lineplot(data=data_to_plot, x='dataset_size', y='avg_time_ms', hue='library', style='epsilon', 
                 markers=True, palette='viridis', dashes=True, 
                 markersize=marker_size, 
                 linewidth=line_width,
                 markeredgewidth=marker_edge_width)
    plt.title(f'Scalability of Performance (Time vs. Dataset Size) for {query_name} Query - Laplace')
    plt.xlabel('Dataset Size (Number of Rows)')
    plt.ylabel('Average Execution Time (ms)')
    plt.xscale('log')
    plt.yscale('log') 
    if unique_dataset_sizes_sorted:
        plt.xticks(unique_dataset_sizes_sorted, dataset_size_tick_labels, rotation=30, ha='right')
    plt.legend(title='Legend', loc='best', fontsize='medium')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, f'scalability_performance_vs_datasize_{query_name.lower()}_laplace.png'))
    plt.show()

print(f"Plotting complete using saved data. Check for .png files in '{plot_output_dir}'.")