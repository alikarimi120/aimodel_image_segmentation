import os
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ConfigLoader:
    def __init__(self, config_path):
        # Load configuration from the provided JSON file
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

    def get_config(self):
        # Return the loaded configuration
        return self.config


class HeatmapPlotter:
    def __init__(self, directory_path, log_file_paths):
        self.file_paths = log_file_paths
        self.directory_path = directory_path
        self.dataframes = [pd.read_csv(file_path, index_col=0) for file_path in log_file_paths]

    def plot_heatmap(self, df, title, filepath, colorful=True):
        # Select only numeric columns
        plot_data = df.select_dtypes(include=[float, int])

        # Separate 'support' columns
        non_support_columns = [col for col in plot_data.columns if 'support' not in col.lower()]

        if colorful:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(plot_data[non_support_columns], annot=True, fmt=".2f", cmap='viridis', cbar=True, ax=ax)
        else:
            # For the second table, do not use heatmap, just plot the values
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.2)
            for key, cell in table.get_celld().items():
                cell.set_edgecolor('black')
                if key[0] == 0 or key[1] == -1:
                    cell.set_facecolor('#d3d3d3')  # Header color
                else:
                    cell.set_facecolor('#f0f8ff')  # Data cell color

        plt.title(title)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    def plot_all_heatmaps(self):
        for i, df in enumerate(self.dataframes):
            title = f'Heatmap of {os.path.basename(self.file_paths[i])}'
            filepath = f'{self.directory_path}/images/id_classification_report{i}.png'
            colorful = i == 0
            self.plot_heatmap(df, title, filepath, colorful=colorful)
            print(f"Heatmap saved for {os.path.basename(self.file_paths[i])}")


if __name__ == "__main__":
    # Usage of the class to read data and plot heatmaps

    config_loader = ConfigLoader('../config/config.json')
    config = config_loader.get_config()
    root_directory_path = config["root_output_path"]

    file_paths = [root_directory_path + '/logss/id_classification_report.csv',
                  root_directory_path + '/logss/id_classification_report2.csv']
    plotter = HeatmapPlotter(root_directory_path, file_paths)
    plotter.plot_all_heatmaps()
