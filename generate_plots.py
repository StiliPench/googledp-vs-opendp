import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# load the combined results
results_filepath_csv = "combined_results.csv"

try:
    plot_df_combined = pd.read_csv(results_filepath_csv)
    print(f"Successfully loaded results from {results_filepath_csv}")
except FileNotFoundError:
    print(f"Error: Results file '{results_filepath_csv}' not found.")
    print("Please run the experiment scripts to generate and save the results.")
    exit()

if plot_df_combined.empty:
    print("Loaded DataFrame is empty. No data to plot.")
    exit()

if 'mechanism' not in plot_df_combined.columns:
    print("Column 'mechanism' not found. Assuming all rows are Laplace.")
    plot_df_combined['mechanism'] = 'Laplace'

plot_df_combined['mechanism'] = plot_df_combined['mechanism'].str.capitalize()


# global plotting setup
sns.set_theme(style="whitegrid", context="talk")
plot_output_dir = "comparison_plots"
os.makedirs(plot_output_dir, exist_ok=True)

marker_size = 23
line_width = 5
marker_edge_width = 1.5

# key epsilons for scalability plots
key_epsilons_plot = [0.1, 1.0, 3.0]

# fataset size tick labels
unique_dataset_sizes_sorted = sorted(plot_df_combined['dataset_size'].unique())
dataset_size_tick_labels = []
for size_val in unique_dataset_sizes_sorted:
    if size_val == 10000:
        dataset_size_tick_labels.append('Small (10k)')
    elif size_val == 100000:
        dataset_size_tick_labels.append('Medium (100k)')
    elif size_val == 1000000:
        dataset_size_tick_labels.append('Large (1M)')
    else:
        dataset_size_tick_labels.append(str(size_val))

# plotting per mechanism 
mechanisms = sorted(plot_df_combined['mechanism'].unique())
print("Mechanisms found in data:", mechanisms)

for mechanism in mechanisms:
    mech_df = plot_df_combined[plot_df_combined['mechanism'] == mechanism]
    if mech_df.empty:
        print(f"No data for mechanism '{mechanism}', skipping.")
        continue

    print(f"\n=== Generating plots for mechanism: {mechanism} ===")

    # ------------------------------
    # Utility: MAE vs Epsilon
    # ------------------------------
    for query_name in ['Count', 'Sum']:
        plt.figure(figsize=(15, 9))
        data_to_plot = mech_df[mech_df['query'] == query_name]
        if data_to_plot.empty:
            print(f"No data for {query_name} query ({mechanism}) to plot MAE vs Epsilon.")
            plt.close()
            continue

        sns.lineplot(
            data=data_to_plot,
            x='epsilon',
            y='mae',
            hue='library',
            style='dataset_name',
            markers=True,
            dashes=True,
            markersize=marker_size,
            linewidth=line_width,
            markeredgewidth=marker_edge_width
        )
        plt.title(f'Utility (MAE vs. Epsilon) for {query_name} Query - {mechanism}')
        plt.xlabel('Epsilon (Privacy Budget)')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.legend(title='Legend', fontsize='medium')
        plt.yscale('log')
        plt.tight_layout()

        fname = f'utility_vs_epsilon_{query_name.lower()}_{mechanism.lower()}.png'
        plt.savefig(os.path.join(plot_output_dir, fname))
        plt.show()

    # ------------------------------
    # Performance: Time vs Epsilon
    # ------------------------------
    for query_name in ['Count', 'Sum']:
        plt.figure(figsize=(15, 9))
        data_to_plot = mech_df[mech_df['query'] == query_name]
        if data_to_plot.empty:
            print(f"No data for {query_name} query ({mechanism}) to plot Time vs Epsilon.")
            plt.close()
            continue

        sns.lineplot(
            data=data_to_plot,
            x='epsilon',
            y='avg_time_ms',
            hue='library',
            style='dataset_name',
            markers=True,
            dashes=True,
            markersize=marker_size,
            linewidth=line_width,
            markeredgewidth=marker_edge_width
        )
        plt.title(f'Performance (Time vs. Epsilon) for {query_name} Query - {mechanism}')
        plt.xlabel('Epsilon (Privacy Budget)')
        plt.ylabel('Average Execution Time (ms)')
        plt.legend(title='Legend', fontsize='medium')
        plt.tight_layout()

        fname = f'performance_vs_epsilon_{query_name.lower()}_{mechanism.lower()}.png'
        plt.savefig(os.path.join(plot_output_dir, fname))
        plt.show()

    # ------------------------------
    # Scalability: Utility vs Dataset Size (key epsilons)
    # ------------------------------
    mech_df_key_eps = mech_df[mech_df['epsilon'].isin(key_epsilons_plot)]

    for query_name in ['Count', 'Sum']:
        plt.figure(figsize=(15, 9))
        data_to_plot = mech_df_key_eps[mech_df_key_eps['query'] == query_name]
        if data_to_plot.empty:
            print(f"No data for {query_name} query ({mechanism}) to plot MAE vs Dataset Size (key epsilons).")
            plt.close()
            continue

        sns.lineplot(
            data=data_to_plot,
            x='dataset_size',
            y='mae',
            hue='library',
            style='epsilon',
            markers=True,
            palette='viridis',
            dashes=True,
            markersize=marker_size,
            linewidth=line_width,
            markeredgewidth=marker_edge_width
        )
        plt.title(
            f'Scalability of Utility (MAE vs. Dataset Size) for {query_name} Query - {mechanism}'
        )
        plt.xlabel('Dataset Size (Number of Rows)')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.xscale('log')
        if unique_dataset_sizes_sorted:
            plt.xticks(unique_dataset_sizes_sorted, dataset_size_tick_labels,
                       rotation=30, ha='right')
        plt.legend(title='Legend', loc='best', fontsize='medium')
        plt.tight_layout()

        fname = f'scalability_utility_vs_datasize_{query_name.lower()}_{mechanism.lower()}.png'
        plt.savefig(os.path.join(plot_output_dir, fname))
        plt.show()

    # ------------------------------
    # Scalability: Performance vs Dataset Size (key epsilons)
    # ------------------------------
    for query_name in ['Count', 'Sum']:
        plt.figure(figsize=(15, 9))
        data_to_plot = mech_df_key_eps[mech_df_key_eps['query'] == query_name]
        if data_to_plot.empty:
            print(f"No data for {query_name} query ({mechanism}) to plot Time vs Dataset Size (key epsilons).")
            plt.close()
            continue

        sns.lineplot(
            data=data_to_plot,
            x='dataset_size',
            y='avg_time_ms',
            hue='library',
            style='epsilon',
            markers=True,
            palette='viridis',
            dashes=True,
            markersize=marker_size,
            linewidth=line_width,
            markeredgewidth=marker_edge_width
        )
        plt.title(
            f'Scalability of Performance (Time vs. Dataset Size) for {query_name} Query - {mechanism}'
        )
        plt.xlabel('Dataset Size (Number of Rows)')
        plt.ylabel('Average Execution Time (ms)')
        plt.xscale('log')
        plt.yscale('log')
        if unique_dataset_sizes_sorted:
            plt.xticks(unique_dataset_sizes_sorted, dataset_size_tick_labels,
                       rotation=30, ha='right')
        plt.legend(title='Legend', loc='best', fontsize='medium')
        plt.tight_layout()

        fname = f'scalability_performance_vs_datasize_{query_name.lower()}_{mechanism.lower()}.png'
        plt.savefig(os.path.join(plot_output_dir, fname))
        plt.show()

print(f"\nPlotting complete using saved data. Check for .png files in '{plot_output_dir}'.")
